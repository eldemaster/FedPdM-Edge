import json
import numpy as np
from common import load_cmapss_raw, USEFUL_SENSORS, set_seed

set_seed(42)
df_raw = load_cmapss_raw()
unique_units = df_raw['unit'].unique()
np.random.shuffle(unique_units)
train_units = unique_units[:-10]  # The 90 units used for training

# Compute global min and max over the healthy part (first 50 cycles)
df_train = df_raw[df_raw['unit'].isin(train_units)].copy()
healthy_stats = df_train.groupby('unit').head(50)
global_min = healthy_stats[USEFUL_SENSORS].min().to_dict()
global_max = healthy_stats[USEFUL_SENSORS].max().to_dict()

with open('global_scale.json', 'w') as f:
    json.dump({'min': global_min, 'max': global_max}, f)
print("Global scale computed and saved.")
