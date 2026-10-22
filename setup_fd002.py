import json
import numpy as np
import pandas as pd
import zipfile

cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
usecols = [i for i in range(26)]
with zipfile.ZipFile('/home/eldemaster/CMAPSSData.zip', 'r') as z:
    with z.open('train_FD002.txt') as f:
        df = pd.read_csv(f, sep=r'\s+', header=None, usecols=usecols, names=cols)

lifespans = df.groupby('unit')['cycle'].max().sort_values()
units_sorted = lifespans.index.tolist()

mid = len(units_sorted) // 2
c1 = units_sorted[:mid]
c2 = units_sorted[mid:]

partition = {
    "1": [int(x) for x in c1],
    "2": [int(x) for x in c2]
}
with open('partition.json', 'w') as f:
    json.dump(partition, f)
print("Created Non-IID partition.json for FD002")
