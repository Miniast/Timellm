import os
import numpy as np
import pandas as pd
from sympy.physics.units import percent
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import json
import bisect


class DatasetCapacity(Dataset):
    def __init__(
            self, root_path, flag='train', size=None, features='M', data_path=None, target=None,
            scale=False, time_enc=0, freq='15min', percent=100
    ):
        target = ['cpu', 'memory']

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.time_enc = time_enc
        self.freq = freq
        self.flag = flag
        self.percent = percent

        self.root_path = root_path
        if flag == 'train':
            self.data_path = 'Train.csv'
        elif flag == 'test':
            self.data_path = 'Test.csv'
        else:
            self.data_path = 'Val.csv'
        self.__read_data__()

        self.enc_in = self.data.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # if self.scale:
        #     # train_data = df_data[border1s[0]:border2s[0]]
        #     train_data = df_data[border['train'][0]:border['train'][1]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # unix ms to datetime
        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date, unit='ms')
        data_stamp = None

        if self.time_enc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.time_enc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # df_raw drop date column
        self.data = df_raw.drop(labels=['date'], axis=1)
        self.data_stamp = data_stamp
        with open(f'{self.root_path}/Data_desc.json', 'r') as f:
            desc = json.load(f)
        self.data_list = desc[self.flag]

    def __getitem__(self, index):
        pos = bisect.bisect_right(self.data_list, index)
        seq_begin = (index + pos * 7) * 24
        seq_end = seq_begin + self.seq_len
        label_begin = seq_end - self.label_len
        label_end = label_begin + self.label_len + self.pred_len

        seq_feature = self.data[seq_begin:seq_end].values
        label_feature = self.data[label_begin:label_end].values
        seq_stamp = self.data_stamp[seq_begin:seq_end]
        label_stamp = self.data_stamp[label_begin:label_end]
        

        return seq_feature, label_feature, seq_stamp, label_stamp
        # feature_id = index // self.tot_len
        # s_begin = index % self.tot_len
        #
        # s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len
        # seq_x = self.data_x[s_begin:s_end, feature_id:feature_id + 1]
        # seq_y = self.data_y[r_begin:r_end, feature_id:feature_id + 1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        #
        # return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.data_list[-1]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
