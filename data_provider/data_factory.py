from data_provider.data_loader import DatasetCapacity
from torch.utils.data import DataLoader

# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'ECL': Dataset_Custom,
#     'Traffic': Dataset_Custom,
#     'Weather': Dataset_Custom,
#     'm4': Dataset_M4,
# }


def data_provider(args, flag):
    # Data = data_dict.get(args.data, DatasetCapacity)
    time_enc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    args.freq = '15min'

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = DatasetCapacity(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        time_enc=time_enc,
        freq=freq,
        percent=percent
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
