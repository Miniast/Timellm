from torch.utils.data import DataLoader
from data_provider_stock.data_loader import DatasetStock


def data_provider(args, data_path, flag):
    time_enc = 0 if args.embed != 'timeF' else 1

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

    data_set = DatasetStock(
        root_path=args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        time_enc=time_enc,
        scale=args.scale,
        freq=freq,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
