import argparse
import torch
from sympy import false
from torch import nn, optim
from torch.optim import lr_scheduler

from data_provider_stock.data_factory import data_provider
from models import TimeLLM

import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, vali

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


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        content = f.read()

    return content


def main():
    device = 'cuda:0'
    setting = 'Stock_Timellm'
    args.scale = False

    train_data, train_loader = data_provider(args, 'Train.csv', 'train')
    test_data, test_loader = data_provider(args, 'Test.csv', 'test')

    args.content = load_prompt('./dataset/prompt_bank/Stock.txt')
    model = TimeLLM.Model(args).float()
    path = os.path.join(args.checkpoints, setting)

    if not os.path.exists(path):
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                # adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)

                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        print("Train loss: {}".format(train_loss))
        # test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_loss)
    #     print(
    #         f'EPOCH: {epoch + 1:03d}, Train Loss: {train_loss:.7f}, Test Loss: {test_loss:.7f}, Test MAE Loss: {test_mae_loss:.7f}'
    #     )
    #
    #     early_stopping(test_loss, model, path)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break
    #
    #     if args.lradj != 'TST':
    #         if args.lradj == 'COS':
    #             scheduler.step()
    #             print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #         else:
    #             if epoch == 0:
    #                 args.learning_rate = model_optim.param_groups[0]['lr']
    #                 print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #             adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
    #
    #     else:
    #         print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    #
    # accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
