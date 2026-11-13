#!/usr/bin/env python3
"""Reproduce multi-dataset C-MAPSS baselines and FL ablations.

Example: python experiments/reproduce.py --data /path/CMAPSSData.zip
The default run uses five paired seeds and writes one self-contained JSON file
per dataset/partition/seed below results/reproducible/.
"""
import argparse
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ['setting1', 'setting2', 'setting3', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
COLS = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
SEEDS = (42, 100, 2024, 7, 777)
WINDOW, CAP = 30, 125


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(17, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Flatten(), nn.Linear(64 * 7, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class CNNLSTM(nn.Module):
    """Small convolutional encoder with a 24-unit recurrent head (~4.9k params)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(17, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(16, 24, batch_first=True)
        self.head = nn.Linear(24, 1)

    def forward(self, x):
        z, _ = self.lstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        return self.head(z[:, -1]).squeeze(-1)


def read_data(path, ds):
    with zipfile.ZipFile(path) as z:
        def frame(name):
            with z.open(name) as f:
                return pd.read_csv(f, sep=r'\s+', header=None, usecols=range(26), names=COLS)
        tr, te = frame(f'train_{ds}.txt'), frame(f'test_{ds}.txt')
        with z.open(f'RUL_{ds}.txt') as f:
            rul = pd.read_csv(f, header=None).iloc[:, 0].to_numpy(dtype=np.float32)
    lo, hi = tr[SENSORS].min(), tr[SENSORS].max()
    for df in (tr, te):
        df[SENSORS] = ((df[SENSORS] - lo) / (hi - lo + 1e-7)).clip(-5, 5)
    def windows(df, train):
        out = {}
        for unit, g in df.groupby('unit', sort=True):
            v = g[SENSORS].to_numpy(dtype=np.float32)
            if train:
                if len(v) >= WINDOW:
                    x = np.lib.stride_tricks.sliding_window_view(v, WINDOW, axis=0).transpose(0, 2, 1).copy()
                    # Compute target from each window's ending cycle.
                    ends = g.cycle.to_numpy()[WINDOW-1:]
                    y = np.minimum(CAP, int(g.cycle.max()) - ends).astype(np.float32)
                    out[int(unit)] = (x, y)
            elif len(v) > 0:
                seq = v[-WINDOW:]
                if len(seq) < WINDOW:
                    # Keep every official test engine in evaluation. Repeat
                    # its first observed row as deterministic left padding.
                    pad = np.repeat(seq[:1], WINDOW-len(seq), axis=0)
                    seq = np.concatenate((pad, seq), axis=0)
                out[int(unit)] = (seq[None], np.array([min(CAP, rul[int(unit)-1])], dtype=np.float32))
        return out
    return windows(tr, True), windows(te, False), tr, te


def nasa(y, p):
    d = np.asarray(p).reshape(-1) - np.asarray(y).reshape(-1)
    return float(np.expm1(np.where(d < 0, -d / 13.0, d / 10.0)).sum())


def metrics(model, data, units):
    xs = np.concatenate([data[u][0] for u in units])
    ys = np.concatenate([data[u][1] for u in units])
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(xs)).cpu().numpy()
    return {'rmse': float(np.sqrt(np.mean((ys-p)**2))), 'nasa': nasa(ys, p)}


def pooled_metrics(models, data, unit_groups):
    ys, preds = [], []
    for model, units in zip(models, unit_groups):
        xs = np.concatenate([data[u][0] for u in units])
        ys.extend(np.concatenate([data[u][1] for u in units]).tolist())
        model.eval()
        with torch.no_grad():
            preds.extend(model(torch.from_numpy(xs)).cpu().numpy().tolist())
    return {'rmse': float(np.sqrt(np.mean((np.asarray(ys)-np.asarray(preds))**2))),
            'nasa': nasa(ys, preds)}


def operating_clusters(frame, k=6):
    """Fit deterministic k-means to cycle-level operating settings."""
    settings = frame[['setting1','setting2','setting3']].to_numpy(dtype=float)
    scale = settings.std(axis=0); scale[scale == 0] = 1.0
    z = (settings-settings.mean(axis=0))/scale
    centers = [z[np.argmin(z[:,0])]]
    for _ in range(1, min(k, len(z))):
        distance = np.min(np.stack([((z-c)**2).sum(axis=1) for c in centers]), axis=0)
        centers.append(z[np.argmax(distance)])
    centers=np.asarray(centers)
    for _ in range(50):
        labels=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        updated=np.stack([z[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(len(centers))])
        if np.allclose(updated,centers): break
        centers=updated
    labels=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
    return centers, settings.mean(axis=0), scale


def window_regime_data(data, centers, mean, scale):
    """Group each engine's windows by the operating condition at window end."""
    grouped={}
    for unit,(x,y) in data.items():
        z=(x[:,-1,:3]-mean)/scale
        labels=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        for regime in np.unique(labels):
            ix=np.flatnonzero(labels==regime)
            key=-(int(unit)*6+int(regime)+1)
            grouped[key]=(x[ix],y[ix])
    return grouped


def decode_regime_key(key):
    value=-int(key)-1
    return value//6, value%6


def partitions(train, seed):
    units = sorted(train)
    rng = np.random.default_rng(seed)
    shuffled = np.array(units); rng.shuffle(shuffled)
    out = {'iid': [list(map(int, shuffled[::2])), list(map(int, shuffled[1::2]))]}
    # Engine-count imbalance holds unit-level examples intact (1:3 engine ratio).
    cut = max(1, len(shuffled) // 4)
    out['engine_imbalance'] = [list(map(int, shuffled[:cut])), list(map(int, shuffled[cut:]))]
    return out


def tensors(data, units):
    x = np.concatenate([data[u][0] for u in units]); y = np.concatenate([data[u][1] for u in units])
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def train(model, data, units, epochs, seed, loss_kind='mse'):
    ds = tensors(data, units)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=128, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(); p = model(x)
            if loss_kind == 'asymmetric':
                d = p-y
                loss = torch.where(d < 0, torch.exp(torch.clamp(-d/13, max=20))-1,
                                   torch.exp(torch.clamp(d/10, max=20))-1).mean()
            else:
                loss = ((p-y)**2).mean()
            loss.backward(); opt.step()
    return model


def clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_variant(name, train_data, test_data, split, test_split, seed, architecture='cnn'):
    make = CNN if architecture == 'cnn' else CNNLSTM
    all_train = sorted(train_data)
    init = make(); initial = clone(init)
    # Every method receives the same nominal ten local epochs of work.
    if name in ('centralized', 'asymmetric', 'cnn_lstm'):
        model = make(); model.load_state_dict(initial)
        loss = 'asymmetric' if name == 'asymmetric' else 'mse'
        model = train(model, train_data, all_train, 10, seed, loss)
        return {'global': metrics(model, test_data, sorted(test_data)),
                'clients': [metrics(model, test_data, us) for us in test_split], 'round_seconds': []}
    if name == 'local_only':
        local = []; models=[]
        for cid, units in enumerate(split):
            model = make(); model.load_state_dict(initial)
            model = train(model, train_data, units, 10, seed+cid)
            local.append(metrics(model, test_data, test_split[cid]))
            models.append(model)
        return {'global': pooled_metrics(models, test_data, test_split), 'clients': local, 'round_seconds': []}
    global_model = make(); global_model.load_state_dict(initial)
    state = clone(global_model); round_seconds = []
    m1 = torch.zeros_like(torch.cat([p.detach().flatten() for p in global_model.parameters()]))
    m2 = torch.zeros_like(m1); step = 0
    for rnd in range(5):
        start = time.perf_counter(); local_states=[]; weights=[]
        for cid, units in enumerate(split):
            local = make(); local.load_state_dict(state)
            if name == 'fedprox':
                local = train_prox(local, train_data, units, 2, seed+rnd+cid, state, mu=0.01)
            else:
                local = train(local, train_data, units, 2, seed+rnd+cid,
                              'asymmetric' if name == 'federated_asymmetric' else 'mse')
            local_states.append(clone(local)); weights.append(len(tensors(train_data, units)))
        total = sum(weights)
        avg = {k: sum(s[k]*w/total for s,w in zip(local_states,weights)) for k in state}
        if name == 'fedadam':
            # Server Adam over pseudo-gradient (global - weighted client average).
            old = torch.cat([v.flatten() for v in state.values() if v.is_floating_point()])
            new = torch.cat([v.flatten() for v in avg.values() if v.is_floating_point()])
            grad = old-new; step += 1; m1 = .9*m1+.1*grad; m2=.99*m2+.01*grad.square()
            update = old - .01*(m1/(1-.9**step))/(torch.sqrt(m2/(1-.99**step))+1e-8)
            offset=0; adjusted={}
            for k,v in avg.items():
                if v.is_floating_point():
                    n=v.numel(); adjusted[k]=update[offset:offset+n].view_as(v); offset+=n
                else: adjusted[k]=v
            avg=adjusted
        state=avg; global_model.load_state_dict(state)
        round_seconds.append(time.perf_counter()-start)
    return {'global': metrics(global_model, test_data, sorted(test_data)),
            'clients': [metrics(global_model, test_data, us) for us in test_split], 'round_seconds': round_seconds}


def train_prox(model, data, units, epochs, seed, reference, mu):
    ds=tensors(data, units); loader=DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); refs={k:v.detach().clone() for k,v in reference.items()}
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            opt.zero_grad(); p=model(x)
            prox=sum((v-refs[k]).pow(2).sum() for k,v in model.named_parameters())
            loss=((p-y)**2).mean()+mu*.5*prox; loss.backward(); opt.step()
    return model


def paired_p(a,b):
    """Exact two-sided sign-flip paired randomization p-value."""
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    if len(d)>20: raise ValueError('exact test is intended for <=20 paired seeds')
    observed=abs(float(d.mean())); means=[abs(float(np.mean(d*np.array(signs)))) for signs in __import__('itertools').product((-1,1),repeat=len(d))]
    return sum(x>=observed-1e-12 for x in means)/len(means)


def main():
    default_out=str(ROOT/'results/reproducible')
    ap=argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--out',default=default_out); ap.add_argument('--datasets',nargs='+',default=['FD001','FD002','FD003','FD004']); ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS)); ap.add_argument('--partitions',nargs='+',default=['iid','regime','engine_imbalance']); ap.add_argument('--threads',type=int,default=1); ap.add_argument('--quick',action='store_true'); ap.add_argument('--summarize-only',action='store_true'); args=ap.parse_args()
    if args.summarize_only:
        summarize(Path(args.out)); return
    if not args.data:
        ap.error('--data is required; dataset archives are not stored in this repository')
    torch.set_num_threads(args.threads)
    if args.quick:
        args.datasets=['FD001']; args.seeds=[42]; args.partitions=['iid']
        if args.out==default_out: args.out=str(ROOT/'results/reproducible/smoke')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for ds in args.datasets:
        tr,te,raw,test_raw=read_data(args.data,ds); splits=partitions(tr,0)
        # Match test clients to the same partition principle. Regime clients
        # receive engines with the corresponding operating settings.
        test_splits = {'iid': [sorted(te)[::2], sorted(te)[1::2]]}
        test_order = np.array(sorted(te)); np.random.default_rng(0).shuffle(test_order)
        cut = max(1, len(test_order)//4)
        test_splits['engine_imbalance'] = [list(map(int,test_order[:cut])),list(map(int,test_order[cut:]))]
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            for partition in args.partitions:
                if partition=='engine_imbalance' and ds not in ('FD002','FD004'):
                    print(f'{ds} has no requested engine-imbalance comparison; skipped',flush=True)
                    continue
                if partition == 'regime' and ds in ('FD002','FD004'):
                    centers,mean,scale=operating_clusters(raw)
                    tr_run=window_regime_data(tr,centers,mean,scale)
                    te_run=window_regime_data(te,centers,mean,scale)
                    order=sorted(range(len(centers)),key=lambda i: tuple(centers[i]))
                    left=set(order[:len(order)//2])
                    split=[[k for k in sorted(tr_run) if (decode_regime_key(k)[1] in left)==side]
                           for side in (True,False)]
                    test_split=[[k for k in sorted(te_run) if (decode_regime_key(k)[1] in left)==side]
                                for side in (True,False)]
                    train_run,test_run=tr_run,te_run
                elif partition in splits:
                    split=splits[partition]
                    test_split=test_splits[partition]
                    train_run,test_run=tr,te
                else:
                    print(f'{ds} has no valid {partition} split; skipped',flush=True)
                    continue
                variants=['centralized','local_only','fedavg']
                if ds in ('FD002','FD004') and partition in ('regime','engine_imbalance'):
                    variants += ['fedprox','fedadam']
                if ds in ('FD002','FD004') and partition=='regime':
                    variants += ['federated_asymmetric','asymmetric','cnn_lstm']
                client_engines=([len({decode_regime_key(k)[0] for k in x}) for x in split] if partition=='regime' else [len(x) for x in split])
                engine_overlap=(len({decode_regime_key(k)[0] for k in split[0]} & {decode_regime_key(k)[0] for k in split[1]}) if partition=='regime' else 0)
                result={'dataset':ds,'partition':partition,'seed':seed,'train_engines':len(tr),'test_engines':len(te),'client_engines':client_engines,
                        'client_train_groups':[len(x) for x in split],'client_test_engines':[len(x) for x in test_split],'client_engine_overlap':engine_overlap,
                        'client_training_windows':([sum(len(train_run[k][1]) for k in x) for x in split]),
                        'protocol':{'window':WINDOW,'rul_cap':CAP,'local_epochs':2,'rounds':5,'central_epochs':10,'batch_size':128,'threads':args.threads,'seeds':list(args.seeds),'scaler':'global min/max fit on training engines','clip':'normalized settings/sensors to [-5,5]','short_test_windows':'left-pad with first observed row to include every official test engine','regime':'for FD002/FD004, group each window by its endpoint regime using six-cluster k-means fitted to cycle-level operating settings','mTLS':False}}
                for name in variants:
                    torch.manual_seed(seed)
                    result[name]=run_variant(name,train_run,test_run,split,test_split,seed, 'lstm' if name=='cnn_lstm' else 'cnn')
                path=out/f'{ds.lower()}_{partition}_seed{seed}.json'; path.write_text(json.dumps(result,indent=2)+'\n')
                print(path, flush=True)
                summarize(out)
    summarize(out)


def summarize(out):
    files=sorted(out.glob('fd*_seed*.json')); grouped={}
    for p in files:
        try: x=json.loads(p.read_text())
        except (ValueError,KeyError): continue
        grouped.setdefault((x['dataset'],x['partition']),[]).append(x)
    rows=[]
    for (ds,part),runs in sorted(grouped.items()):
        for method in ('centralized','local_only','fedavg','fedprox','fedadam','federated_asymmetric','asymmetric','cnn_lstm'):
            if any(method not in r for r in runs):
                continue
            client_count=max((len(r[method].get('clients',[])) for r in runs),default=0)
            scopes=[('global',None)]+[('client',i) for i in range(client_count)]
            for scope,client_id in scopes:
                for metric in ('rmse','nasa'):
                    vals=[]
                    for r in runs:
                        measured=(r[method]['global'] if scope=='global' else
                                  r[method].get('clients',[])[client_id] if client_id<len(r[method].get('clients',[])) else None)
                        if measured is not None:
                            vals.append(measured[metric])
                    if vals:
                        rows.append({'dataset':ds,'partition':part,'method':method,'scope':scope,'client_id':client_id if client_id is not None else '',
                                     'metric':metric,'n':len(vals),'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0})
        if len(runs)>1:
            comparisons=(('fedavg','centralized'),('fedavg','local_only'),('fedavg','fedprox'),
                         ('fedavg','fedadam'),('fedavg','federated_asymmetric'),
                         ('centralized','asymmetric'),('centralized','cnn_lstm'))
            for reference,method in comparisons:
                if any(reference not in r or method not in r for r in runs):
                    continue
                for metric in ('rmse','nasa'):
                    a=[r[reference]['global'][metric] for r in runs]; b=[r[method]['global'][metric] for r in runs]
                    rows.append({'dataset':ds,'partition':part,'comparison':f'{reference}_vs_{method}','scope':'global','metric':metric,'paired_sign_flip_p':paired_p(a,b),'n':len(a)})
    if rows:
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)


if __name__=='__main__': main()
