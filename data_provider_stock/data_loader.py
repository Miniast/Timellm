import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


class DatasetStock(Dataset):
    def __init__(
            self, root_path, flag, size, features, data_path, scale, time_enc, freq
    ):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.scale = scale
        self.time_enc = time_enc
        self.freq = freq
        self.flag = flag

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data.shape[-1]
        self.len = len(self.data) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(f'{self.root_path}/{self.data_path}')

        '''
            df_raw.columns: ['date', 'open', 'high', 'low', 'close']
            date,open,high,low,close,volume
            2016-01-04,30.57,30.57,28.63,28.78
        '''

        data_cols = df_raw.columns[1:-1]
        df_data = df_raw[data_cols]

        sample_num = df_data.shape[0] - self.seq_len - self.pred_len + 1
        if self.scale:
            train_data = df_data[:sample_num]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = None
        if self.time_enc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feature_id = index // self.len
        index = index % self.len
        seq_begin = index
        seq_end = index + self.seq_len
        res_begin = seq_end - self.label_len
        res_end = res_begin + self.label_len + self.pred_len

        # 选择 seq begin 到 seq end 的对应列
        seq = self.data[seq_begin:seq_end, feature_id]
        res = self.data[res_begin:res_end, feature_id]
        seq_stamp = self.data_stamp[seq_begin:seq_end]
        res_stamp = self.data_stamp[res_begin:res_end]

        return seq, res, seq_stamp, res_stamp

    def __len__(self):
        return self.len * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
