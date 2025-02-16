import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--root_path', type=str, default='./dataset/Capacity', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Train.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--freq', type=str, default='15min',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

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
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=32)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin,
                          gradient_accumulation_steps=8)


def smape_loss(pred, true):
    return torch.mean(torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-6))


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        content = f.read()

    return content


def validate(args, accelerator, model, val_data, val_loader, loss_func, mae_loss_func, mse_loss_func):
    total_loss = []
    total_mae_loss = []
    total_mse_loss = []

    model.eval()
    with torch.no_grad():
        for i, (seq, tar, seq_timestamp, tar_timestamp) in tqdm(
                enumerate(val_loader), disable=not accelerator.is_local_main_process
        ):
            seq = seq.float().to(accelerator.device)
            tar = tar.float().to(accelerator.device)
            seq_timestamp = seq_timestamp.float().to(accelerator.device)
            tar_timestamp = tar_timestamp.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(tar[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([tar[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)[0]
                    else:
                        outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)
            else:
                if args.output_attention:
                    outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)[0]
                else:
                    outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)

            outputs = outputs[:, -args.pred_len:, :]
            tar = tar[:, -args.pred_len:, :].to(accelerator.device)

            pred = outputs.detach().squeeze()
            true = tar.detach().squeeze()

            loss = loss_func(pred, true)
            mae_loss = mae_loss_func(pred, true)
            mse_loss = mse_loss_func(pred, true)

            # if loss > 50 or mae_loss > 5:
            #     print('Loss:', loss.item(), 'MAE:', mae_loss.item())
            #     # save seq, pred, true as json
            #     seq = seq.cpu().numpy().tolist()
            #     pred = pred.cpu().numpy().tolist()
            #     true = true.cpu().numpy().tolist()
            #     with open(f'validation/model_1/{i}.json', 'w') as f:
            #         json.dump({'seq': seq, 'pred': pred, 'true': true}, f)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_mse_loss.append(mse_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_mse_loss = np.average(total_mse_loss)

    model.train()
    return total_loss, total_mae_loss, total_mse_loss


def main():
    setting = "Capacity_Timellm"
    args.scale = True

    train_data, train_loader = data_provider(args, 'Train.csv', 'train')
    val_data, val_loader = data_provider(args, 'Val.csv', 'val')

    args.content = load_prompt('./dataset/prompt_bank/Capacity.txt')
    model = TimeLLM.Model(args).float()
    path = os.path.join(args.checkpoints, setting)  # unique checkpoint saving path

    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

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

    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    train_loader, val_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, val_loader, model, model_optim, scheduler
    )

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (seq, tar, seq_timestamp, tar_timestamp) in tqdm(
                enumerate(train_loader), disable=not accelerator.is_local_main_process
        ):
            iter_count += 1
            model_optim.zero_grad()
            # seq is (B, 96, 3), tar is (B, 144, 3), seq_timestamp is (B, 96, 5), tar_timestamp is (B, 144, 5)
            seq = seq.float().to(accelerator.device)
            tar = tar.float().to(accelerator.device)
            seq_timestamp = seq_timestamp.float().to(accelerator.device)
            tar_timestamp = tar_timestamp.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(tar[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([tar[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)[0]
                    else:
                        outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    tar = tar[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = smape_loss(outputs, tar)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)[0]
                else:
                    outputs = model(seq, seq_timestamp, dec_inp, tar_timestamp)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                tar = tar[:, -args.pred_len:, f_dim:]
                # print(outputs, tar)
                # exit(2)

                loss = smape_loss(outputs, tar)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                if accelerator.is_local_main_process:
                    accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                if accelerator.is_local_main_process:
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        if accelerator.is_local_main_process:
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        val_loss, val_mae_loss, val_mse_loss = validate(
            args, accelerator, model, val_data, val_loader, smape_loss, mae_loss, mse_loss
        )

        if accelerator.is_local_main_process:
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} MAE Loss: {3:.7f}".format(
                    epoch + 1, train_loss, val_loss, val_mae_loss))

        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.wait_for_everyone()
    # if accelerator.is_local_main_process:
    #     path = './checkpoints'  # unique checkpoint saving path
    #     del_files(path)  # delete checkpoint files
    #     accelerator.print('success delete checkpoints')


if __name__ == '__main__':
    main()
