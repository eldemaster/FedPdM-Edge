#!/usr/bin/env python3
"""Measure one-client CPU training cost and model-payload size on an edge node.

Run on each Pi with the same dataset archive and environment used by the study:
python experiments/profile_edge.py --data /path/CMAPSSData.zip --dataset FD002 \
  --partition regime --client 0 --output results/edge/rpi_fd002.json

Communication is the exact serialized tensor payload estimate for one global
model downlink plus one client update uplink. It excludes gRPC/TLS framing;
this repository's Flower launcher does not configure TLS.
"""
import argparse
import json
import platform
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reproduce import CAP, COLS, SENSORS, WINDOW, CNN, partitions, train, operating_clusters


def load_client(data_path, dataset, partition, client):
    with zipfile.ZipFile(data_path) as z:
        with z.open(f'train_{dataset}.txt') as f:
            frame = pd.read_csv(f, sep=r'\s+', header=None, usecols=range(26), names=COLS)
    lo, hi = frame[SENSORS].min(), frame[SENSORS].max()
    frame[SENSORS] = ((frame[SENSORS]-lo)/(hi-lo+1e-7)).clip(-5,5)
    units = sorted(frame.unit.unique().astype(int))
    selected_regimes=None
    if partition=='regime':
        if dataset not in ('FD002','FD004'):
            raise ValueError('regime partition is defined only for FD002 and FD004')
        centers,mean,scale=operating_clusters(frame)
        order=sorted(range(len(centers)),key=lambda i:tuple(centers[i]))
        selected_regimes=set(order[:len(order)//2] if client==0 else order[len(order)//2:])
        chosen=set(units)
    else:
        split = partitions({u: None for u in units}, 0)[partition]
        chosen = set(split[client])
        frame = frame[frame.unit.isin(chosen)]
    data = {}
    for unit, group in frame.groupby('unit', sort=True):
        values = group[SENSORS].to_numpy(dtype=np.float32)
        if len(values) < WINDOW:
            continue
        x = np.lib.stride_tricks.sliding_window_view(values, WINDOW, axis=0).transpose(0,2,1).copy()
        ends = group.cycle.to_numpy()[WINDOW-1:]
        y = np.minimum(CAP, int(group.cycle.max())-ends).astype(np.float32)
        if selected_regimes is not None:
            z=(x[:,-1,:3]-mean)/scale
            labels=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            keep=np.isin(labels,list(selected_regimes))
            x,y=x[keep],y[keep]
            if len(y)==0: continue
        data[int(unit)] = (x,y)
    return data, len({u for u in data})


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if platform.system() == 'Darwin' else 1024))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--dataset',choices=['FD001','FD002','FD003','FD004'],default='FD002'); ap.add_argument('--partition',choices=['iid','regime','engine_imbalance'],default='regime'); ap.add_argument('--client',type=int,choices=[0,1],default=0); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--output',required=True); args=ap.parse_args()
    torch.set_num_threads(1)
    data, engine_count=load_client(args.data,args.dataset,args.partition,args.client)
    sample=CNN(); param_bytes=sum(p.numel()*p.element_size() for p in sample.parameters())
    durations=[]
    for rnd in range(args.rounds):
        torch.manual_seed(42+rnd); model=CNN(); start=time.perf_counter()
        train(model,data,sorted(data),args.epochs,42+rnd)
        durations.append(time.perf_counter()-start)
    payload=2*param_bytes
    record={'device':platform.node(),'system':platform.platform(),'processor':platform.processor(),'python':platform.python_version(),'pytorch':torch.__version__,
            'dataset':args.dataset,'partition':args.partition,'client':args.client,'client_engines':engine_count,'training_windows':sum(len(v[1]) for v in data.values()),
            'rounds':args.rounds,'local_epochs':args.epochs,'model':'CNN','train_seconds_per_round':durations,
            'train_seconds_mean':float(np.mean(durations)),'train_seconds_std':float(np.std(durations,ddof=1)) if len(durations)>1 else 0.0,
            'peak_process_rss_bytes':peak_rss_bytes(),'model_tensor_bytes':param_bytes,'communication_tensor_payload_bytes_per_round':payload,
            'communication_definition':'model parameters downlink plus updated parameters uplink; excludes gRPC/TLS framing; payload is measured from tensor dtypes/shapes',
            'mTLS':False}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
