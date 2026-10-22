import pandas as pd
from common import load_cmapss_raw
df = load_cmapss_raw('train_FD002.txt')
lifespans = df.groupby('unit')['cycle'].max()
print("Lifespan distribution:")
print(lifespans.describe())
