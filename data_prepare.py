import os
import pandas as pd
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)

files = os.listdir('./dataset/Data/')
dfs_train = []
dfs_val = []
train_list = []
val_list = []
train_pos = 0
val_pos = 0
for file in files:
    df_now = pd.read_csv(f'./dataset/Data/{file}')
    df_now['device'] = len(train_list)
    # df_now records调整为24的倍数
    records = df_now.shape[0]
    seqs = records // 24
    train_seqs = int(seqs * 0.7)
    val_seqs = seqs - train_seqs
    df_train_now = df_now.iloc[:train_seqs * 24, :]
    df_val_now = df_now.iloc[train_seqs * 24:train_seqs * 24 + val_seqs * 24, :]
    dfs_train.append(df_train_now)
    dfs_val.append(df_val_now)
    train_list.append(train_seqs - 7 + train_pos)
    val_list.append(val_seqs - 7 + val_pos)
    train_pos += train_seqs - 7
    val_pos += val_seqs - 7

# df = pd.concat(dfs)
# df = df.reset_index(drop=True)
# df = df[['date', 'device', 'cpu', 'memory']]
# df.to_csv('./dataset/Total.csv', index=False)
df_train = pd.concat(dfs_train)
df_train = df_train.reset_index(drop=True)
df_train = df_train[['date', 'device', 'cpu', 'memory']]
df_train.to_csv('./dataset/Train.csv', index=False)
df_val = pd.concat(dfs_val)
df_val = df_val.reset_index(drop=True)
df_val = df_val[['date', 'device', 'cpu', 'memory']]
df_val.to_csv('./dataset/Val.csv', index=False)
with open('./dataset/Data_desc.json', 'w') as f:
    json.dump({'train': train_list, 'val': val_list}, f)
