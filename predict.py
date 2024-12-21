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
#-------------------- 配置参数 ----------------------
parser = argparse.ArgumentParser(description='Validation Only')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096',
                    help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args(args=[])

args.content = load_content(args)

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

def smape(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true)))

def vali(args, model, vali_data, vali_loader, criterion, mae_metric):
    print('Validation...')
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to('cuda:0')
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to('cuda:0')
            batch_y_mark = batch_y_mark.float().to('cuda:0')

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to('cuda:0')

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
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to('cuda:0')

            pred = outputs.detach().squeeze()
            true = batch_y.detach().squeeze()

            # loss = criterion(pred, true)
            # mae_loss = mae_metric(pred, true)
            
            # if loss > 50 or mae_loss > 5:
            #     print('Loss:', loss.item(), 'MAE:', mae_loss.item())
            #     # save batch_x, pred, true as json
            #     batch_x = batch_x.cpu().numpy().tolist()
            #     pred = pred.cpu().numpy().tolist()
            #     true = true.cpu().numpy().tolist()
            #     with open(f'validation/model_1/{i}.json', 'w') as f:
            #         json.dump({'batch_x': batch_x, 'pred': pred, 'true': true}, f)

            # total_loss.append(loss.item())
            # total_mae_loss.append(mae_loss.item())
            
            # cpu_loss = mae_metric(pred[:, 1], true[:, 1]) / true[:, 1].mean()
            # memory_loss = mae_metric(pred[:, 2], true[:, 2]) / true[:, 2].mean()

            cpu_loss = smape(pred[:, 1], true[:, 1])
            memory_loss = smape(pred[:, 2], true[:, 2])
            if cpu_loss > 0.2 or memory_loss > 0.2:
                batch_x = batch_x.cpu().numpy().tolist()
                pred = pred.cpu().numpy().tolist()
                true = true.cpu().numpy().tolist()
                with open(f'validation/model_1/{i}.json', 'w') as f:
                    json.dump({'batch_x': batch_x, 'pred': pred, 'true': true}, f)
            print('CPU Loss:', cpu_loss.item(), 'Memory Loss:', memory_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss




if args.model == 'Autoformer':
    model = Autoformer.Model(args).float()
elif args.model == 'DLinear':
    model = DLinear.Model(args).float()
else:
    model = TimeLLM.Model(args).float()

# model_path = os.path.join(args.checkpoints, "result-" + args.model_comment, 'checkpoint')
model_path = 'checkpoints/model_1/checkpoint'

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to('cuda:0')
model.eval()

criterion = nn.MSELoss()
mae_metric = nn.L1Loss()
vali_data, vali_loader = data_provider(args, flag='val')
vali_loss, vali_mae_loss = vali(args, model, vali_data, vali_loader, criterion, mae_metric)

print(f"Validation Loss: {vali_loss:.6f}, Validation MAE: {vali_mae_loss:.6f}")
