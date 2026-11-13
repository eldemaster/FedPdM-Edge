import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn

# Added setting1, setting2, setting3 for heterogeneous operating conditions
USEFUL_SENSORS = ['setting1', 'setting2', 'setting3', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQ_LEN = 30
MAX_RUL = 125

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cmapss_zip_path():
    """Locate CMAPSSData.zip: $CMAPSS_ZIP, then ./, ~/ (no hard-coded user path)."""
    cands = [os.environ.get("CMAPSS_ZIP"), "CMAPSSData.zip",
             os.path.expanduser("~/CMAPSSData.zip")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("CMAPSSData.zip not found; set CMAPSS_ZIP=/path/to/CMAPSSData.zip")

def load_cmapss_raw(filename):
    import zipfile
    cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    usecols = [i for i in range(26)]
    with zipfile.ZipFile(cmapss_zip_path(), 'r') as z:
        with z.open(filename) as f:
            df = pd.read_csv(f, sep=r'\s+', header=None, usecols=usecols, names=cols)
    return df

class CmapssRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels is now 17
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
        # Short test engines used to be silently dropped (biasing the metric to 
        # longer-lived engines). Left-pad by repeating the first reading instead.
        pad = np.repeat(data[:1], SEQ_LEN - len(data), axis=0)
        data = np.concatenate([pad, data], axis=0)
    seq = data[-SEQ_LEN:]
    rul = min(MAX_RUL, true_rul)
    return seq, rul

def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0), np.exp(d / 10.0)) - 1))


def late_prediction_rate(y_true, y_pred):
    """Fraction of predictions that overestimate RUL (late failure prediction: y_pred > y_true)."""
    return float(np.mean(np.asarray(y_pred, dtype=np.float64) > np.asarray(y_true, dtype=np.float64)))


class AsymmetricRULLoss(nn.Module):
    """Asymmetric quadratic loss for Remaining Useful Life (RUL) regression.

    Aligns local and federated optimization with the safety-critical NASA
    Asymmetric Scoring Function (``nasa_score``), which penalizes late failure
    predictions (``y_pred > y_true``, i.e. overestimating RUL and risking
    unpredicted failure) exponentially more heavily than conservative early
    predictions (``y_pred < y_true``).

    Unlike the raw exponential NASA score—whose gradients grow as
    ``exp(|d|/10)`` and explode on large early-training errors—this loss uses
    a smooth, differentiable asymmetric squared-error surrogate:

        L(d) = alpha_late  * d^2   if d >= 0  (late prediction: y_pred > y_true)
             = alpha_early * d^2   if d <  0  (early prediction: y_pred < y_true)

    where ``d = y_pred - y_true``. When ``alpha_late > alpha_early > 0``,
    overestimating RUL is penalized strictly more than equal-magnitude
    underestimation, while gradients remain bounded linearly in ``|d|`` and
    continuous at ``d = 0``.
    """

    def __init__(self, alpha_late: float = 2.5, alpha_early: float = 0.5):
        super().__init__()
        if alpha_late <= 0 or alpha_early <= 0:
            raise ValueError("alpha_late and alpha_early must be positive")
        if alpha_late <= alpha_early:
            raise ValueError("alpha_late must be strictly greater than alpha_early")
        self.alpha_late = float(alpha_late)
        self.alpha_early = float(alpha_early)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        d = y_pred - y_true
        weights = torch.where(d >= 0, self.alpha_late, self.alpha_early)
        return torch.mean(weights * (d ** 2))


def get_loss_fn(name: str = "mse", **kwargs) -> nn.Module:
    """Return a PyTorch loss module by name (``'mse'`` or ``'asymmetric'``)."""
    key = (name or "mse").strip().lower()
    if key == "mse":
        return nn.MSELoss(**kwargs)
    if key in ("asymmetric", "asym", "asymmetric_rul"):
        return AsymmetricRULLoss(**kwargs)
    raise ValueError(f"Unknown loss function {name!r}; expected 'mse' or 'asymmetric'")


def load_rul(filename):
    import zipfile
    with zipfile.ZipFile(cmapss_zip_path(), 'r') as z:
        with z.open(filename) as f:
            return pd.read_csv(f, header=None).values.flatten()


def apply_scale(df, scales):
    mn = pd.Series(scales['min']); mx = pd.Series(scales['max'])
    df = df.copy()
    df[USEFUL_SENSORS] = (df[USEFUL_SENSORS] - mn) / (mx - mn + 1e-7)
    return df


def build_test_set(scales, subset='FD002'):
    """Official test set: last window of every engine (all engines, short ones padded)."""
    df = apply_scale(load_cmapss_raw(f'test_{subset}.txt'), scales)
    ruls = load_rul(f'RUL_{subset}.txt')
    X, y = [], []
    for u, g in df.groupby('unit'):
        x, r = extract_last_sequence(g, ruls[int(u) - 1])
        X.append(x); y.append(r)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------- shared training
import json
from torch.utils.data import DataLoader, TensorDataset


def load_scales(path='global_scale.json'):
    with open(path) as f:
        return json.load(f)


def load_partition(path='partition.json'):
    with open(path) as f:
        return {k: [int(u) for u in v] for k, v in json.load(f).items()}


def make_xy(df_scaled, units):
    X, y = [], []
    for u, g in df_scaled[df_scaled['unit'].isin(units)].groupby('unit'):
        xs, ys = extract_supervised_sequences(g)
        X.extend(xs); y.extend(ys)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_loader(X, y, batch_size=64, seed=0, shuffle=True):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle, generator=g)


def split_units(units, val_frac=0.1, seed=0):
    """Unit-level (not window-level) split so val windows never share an engine with train."""
    rng = np.random.RandomState(seed)
    u = np.array(sorted(units)); rng.shuffle(u)
    k = max(1, int(round(len(u) * val_frac)))
    return [int(x) for x in u[k:]], [int(x) for x in u[:k]]


def train_epochs(model, loader, epochs, lr=1e-3, mu=0.0, global_params=None, optimizer=None, loss_fn=None):
    """Local training. mu>0 adds the FedProx term mu/2*||w - w_global||^2."""
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn is None:
        crit = nn.MSELoss()
    elif isinstance(loss_fn, nn.Module):
        crit = loss_fn
    else:
        crit = get_loss_fn(loss_fn)
    model.train()
    if mu > 0:
        ref = [p.detach().clone() for p in global_params]
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            if mu > 0:
                loss = loss + 0.5 * mu * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
            loss.backward()
            opt.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X)).numpy()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
