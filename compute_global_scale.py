import json
import numpy as np
from common import load_cmapss_raw, USEFUL_SENSORS, extract_supervised_sequences

# Min/max over the first 50 cycles of every *training* engine (healthy regime). Test data is never used.
df_raw = load_cmapss_raw('train_FD002.txt')
head = df_raw.groupby('unit').head(50)[USEFUL_SENSORS]
all_y = []
for u, group in df_raw.groupby('unit'):
    _, y = extract_supervised_sequences(group)
    all_y.extend(y)

with open('global_scale.json', 'w') as f:
    json.dump({'min': head.min().to_dict(), 'max': head.max().to_dict(),
               'train_mean_rul': float(np.mean(all_y))}, f)
