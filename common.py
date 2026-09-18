import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn

USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQ_LEN = 30
MAX_RUL = 125

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def load_cmapss_raw(filename):
    import zipfile
    cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    usecols = [i for i in range(26)]
    with zipfile.ZipFile('/home/eldemaster/CMAPSSData.zip', 'r') as z:
        with z.open(filename) as f:
            df = pd.read_csv(f, sep=r'\s+', header=None, usecols=usecols, names=cols)
    return df

class CmapssRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=len(USEFUL_SENSORS), out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def extract_supervised_sequences(df_group):
    data = df_group[USEFUL_SENSORS].values
    cycles = df_group['cycle'].values
    max_cycle = cycles.max()
    X, y = [], []
    for i in range(len(data) - SEQ_LEN + 1):
        seq = data[i:i+SEQ_LEN]
        rul = min(MAX_RUL, max_cycle - cycles[i + SEQ_LEN - 1])
        X.append(seq)
        y.append(rul)
    return X, y

def extract_last_sequence(df_group, true_rul):
    data = df_group[USEFUL_SENSORS].values
    if len(data) < SEQ_LEN:
        return None, None
    seq = data[-SEQ_LEN:]
    rul = min(MAX_RUL, true_rul)
    return seq, rul

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1
        else:
            score += np.exp(diff / 10.0) - 1
    return score
