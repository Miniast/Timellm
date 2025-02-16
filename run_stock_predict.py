import argparse
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn
from tqdm import tqdm

from data_provider_stock.data_factory import data_provider
from models import TimeLLM

import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
# data loader
parser.add_argument('--root_path', type=str, default='./dataset/Stock', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Train.csv', help='data file')

parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='D',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096',
                    help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=32)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        content = f.read()

    return content


def smape(pred, true):
    return torch.mean(torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-6))


def validate(args, accelerator, model, test_data, test_loader, criterion, mae_loss_func):
    total_loss = []
    total_mae_loss = []
    total_smape_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

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

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(accelerator.device)

            pred = outputs.detach().squeeze()
            true = batch_y.detach().squeeze()

            loss = criterion(pred, true)
            mae_loss = mae_loss_func(pred, true)
            smape_loss = smape(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_smape_loss.append(smape_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_smape_loss = np.average(total_smape_loss)
    return total_loss, total_mae_loss, total_smape_loss


def main():
    setting = 'Stock_Timellm'
    args.scale = False
    test_data, test_loader = data_provider(args, 'Test.csv', 'test')

    args.content = load_prompt('./dataset/prompt_bank/Stock.txt')
    model_path = f'checkpoints/Stock_Timellm/checkpoint'
    model = TimeLLM.Model(args).float()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()

    test_loader, model = accelerator.prepare(
        test_loader, model
    )

    model.to(accelerator.device)
    model.eval()

    test_data, test_loader = data_provider(args, 'Test.csv', 'test')
    test_loss, test_mae_loss, test_smape_loss = validate(args, accelerator, model, test_data, test_loader, criterion,
                                                         mae_loss)
    if accelerator.is_main_process:
        print(f'Test Loss: {test_loss}, Test MAE Loss: {test_mae_loss}, Test SMAPE Loss: {test_smape_loss}')

    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
