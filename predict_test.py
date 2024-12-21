import argparse
import torch
from torch import nn
import numpy as np
import random
import os
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content
import json

# ===== 新增的导入 =====
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

parser = argparse.ArgumentParser(description='Validation Only')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model_id', type=str, default='test')
parser.add_argument('--model_comment', type=str, default='none')
parser.add_argument('--model', type=str, default='Autoformer')
parser.add_argument('--seed', type=int, default=2021)

# data loader
parser.add_argument('--data', type=str, default='ETTm1')
parser.add_argument('--root_path', type=str, default='./dataset')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--loader', type=str, default='modal')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

# model define
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--prompt_domain', type=int, default=1)
parser.add_argument('--llm_model', type=str, default='LLAMA')
parser.add_argument('--llm_dim', type=int, default='4096')

# optimization
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--align_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--pct_start', type=float, default=0.2)
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args(args=[])
args.content = load_content(args)

# 覆盖参数以与训练时的一致
args.task_name = 'long_term_forecast'
args.is_training = 1
args.root_path = './dataset/'
args.model_id = 'Train'
args.model = 'TimeLLM'
args.data = 'Train'
args.features = 'M'
args.seq_len = 96
args.label_len = 48
args.pred_len = 96
args.factor = 3
args.enc_in = 7
args.dec_in = 7
args.c_out = 7
args.des = 'Exp'
args.d_model = 32
args.d_ff = 128
args.batch_size = 1
args.llm_layers = 32
args.model_comment = 'TimeLLM'
args.target = ['cpu', 'memory']

# ===== 新增accelerate配置 =====
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    accelerator.print('Validation...')
    total_cpu_loss = []
    total_memory_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :].to(accelerator.device), dec_inp], dim=1).float()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            cpu_loss = mae_metric(pred[:, :, 1], true[:, :, 1]) / (true[:, :, 1].mean() + 1e-9)
            memory_loss = mae_metric(pred[:, :, 2], true[:, :, 2]) / (true[:, :, 2].mean() + 1e-9)

            if cpu_loss > 0.2 or memory_loss > 0.2:
                accelerator.print('CPU Loss:', cpu_loss.item(), 'Memory Loss:', memory_loss.item())
                batch_x_cpu = batch_x.cpu().numpy().tolist()
                pred_cpu = pred.cpu().numpy().tolist()
                true_cpu = true.cpu().numpy().tolist()
                with open(f'validation/model_1/{i}.json', 'w') as f:
                    json.dump({'batch_x': batch_x_cpu, 'pred': pred_cpu, 'true': true_cpu}, f)

            # 如果需要统计MAE和MSE，可取消注释
            # loss = criterion(pred, true)
            # mae_loss = mae_metric(pred, true)
            # total_loss.append(loss.item())
            # total_mae_loss.append(mae_loss.item())
            total_cpu_loss.append(cpu_loss.item())
            total_memory_loss.append(memory_loss.item())

            # accelerator.print('CPU Loss:', cpu_loss.item(), 'Memory Loss:', memory_loss.item())
    if len(total_loss) > 0:
        total_loss = np.average(total_loss)
        total_mae_loss = np.average(total_mae_loss)
    else:
        total_loss = 0.0
        total_mae_loss = 0.0
    return total_loss, total_mae_loss


if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

model_path = 'checkpoints/model_1/checkpoint'
model.load_state_dict(torch.load(model_path, map_location='cpu'))

# 使用accelerator.device移动模型
model.to(accelerator.device)

criterion = nn.MSELoss()
mae_metric = nn.L1Loss()
vali_data, vali_loader = data_provider(args, flag='val')

# 使用accelerate包装验证数据
vali_loader, model = accelerator.prepare(vali_loader, model)

vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)

accelerator.print(f"Validation Loss: {vali_loss:.6f}, Validation MAE: {vali_mae_loss:.6f}")
