import json
import numpy as np
from common import load_cmapss_raw, USEFUL_SENSORS, set_seed, extract_supervised_sequences

set_seed(42)
df_raw = load_cmapss_raw('train_FD002.txt')
unique_units = df_raw['unit'].unique()
np.random.shuffle(unique_units)

global_min = df_raw.groupby('unit').head(50)[USEFUL_SENSORS].min().to_dict()
global_max = df_raw.groupby('unit').head(50)[USEFUL_SENSORS].max().to_dict()

all_y = []
for u, group in df_raw.groupby('unit'):
    _, y = extract_supervised_sequences(group)
    all_y.extend(y)

with open('global_scale.json', 'w') as f:
    json.dump({
        'min': global_min, 
        'max': global_max,
        'train_mean_rul': float(np.mean(all_y))
    }, f)

# P16: Write partition.json
