import argparse
from contextlib import contextmanager
import inspect
import math
import os
from pathlib import Path
import random
import ssl
import sys
import time
import resource
from collections import OrderedDict
from types import SimpleNamespace

import flwr as fl
import numpy as np
import torch

from common import (CmapssRegressor, USEFUL_SENSORS, set_seed, load_cmapss_raw, load_scales,
                    load_partition, apply_scale, make_xy, make_loader, split_units,
                    train_epochs, predict, rmse, get_loss_fn)


def validate_dp(dp_clip, dp_noise_mult):
    if dp_clip is not None and (not math.isfinite(dp_clip) or dp_clip <= 0):
        raise ValueError("dp_clip must be finite and > 0")
    if not math.isfinite(dp_noise_mult) or dp_noise_mult < 0:
        raise ValueError("dp_noise_mult must be finite and >= 0")
    if dp_noise_mult > 0 and dp_clip is None:
        raise ValueError("dp_noise_mult requires dp_clip")
    if dp_clip is not None and not math.isfinite(dp_clip * dp_noise_mult):
        raise ValueError("dp_clip * dp_noise_mult must be finite")


def transport_certificates(ap, args):
    paths = (args.ca_cert, args.cert, args.key)
    if args.insecure:
        if any(paths):
            ap.error("--insecure conflicts with certificate options/environment variables")
        print("WARNING: INSECURE plaintext Flower transport; local development only", file=sys.stderr)
        return None
    if not args.ca_cert or bool(args.cert) != bool(args.key):
        ap.error("TLS requires --ca-cert / FL_CA_CERT; client --cert and --key must be supplied together "
                 "(FL_CLIENT_CERT/FL_CLIENT_KEY); use --insecure only for local development")
    try:
        ctx = ssl.create_default_context(cafile=args.ca_cert)
        if args.cert:
            ctx.load_cert_chain(args.cert, args.key)
        return tuple(Path(p).read_bytes() if p else None for p in paths)
    except (OSError, ssl.SSLError) as exc:
        ap.error(f"Invalid TLS credentials: {exc}")


@contextmanager
def flower_client_credentials(certificates):
    """Supply client key/cert through Flower 1.22's CA-only gRPC factory.

    Keep start_client(root_certificates=..., insecure=False) and scope the
    adapter to Flower's module reference, without changing grpc globally.
    """
    if certificates is None or certificates[1] is None:
        yield
        return
    import flwr.common.grpc as flower_grpc
    if ("root_certificates" not in inspect.signature(fl.client.start_client).parameters
            or "grpc.ssl_channel_credentials(" not in inspect.getsource(flower_grpc.create_channel)):
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation")
    original = flower_grpc.grpc
    ca, cert, key = certificates

    def credentials(root_certificates=None, private_key=None, certificate_chain=None):
        if root_certificates != ca or private_key is not None or certificate_chain is not None:
            raise RuntimeError("Flower supplied unexpected TLS credentials")
        return original.ssl_channel_credentials(root_certificates=ca, private_key=key,
                                                certificate_chain=cert)

    flower_grpc.grpc = SimpleNamespace(**{**vars(original), "ssl_channel_credentials": credentials})
    try:
        yield
    finally:
        flower_grpc.grpc = original


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


class SupervisedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_x, val_y, lr=1e-3, loss_fn=None,
                 dp_clip=None, dp_noise_mult=0.0, tls_enabled=False, mtls_enabled=False):
        validate_dp(dp_clip, dp_noise_mult)
        self.model = model
        self.train_loader = train_loader
        self.val_x, self.val_y = val_x, val_y
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn if loss_fn is not None else get_loss_fn("mse")
        self.dp_clip, self.dp_noise_mult = dp_clip, dp_noise_mult
        self.tls_enabled, self.mtls_enabled = tls_enabled, mtls_enabled
        # Independent OS randomness: the reproducible training seed must not
        # make privacy noise predictable to the server or repeat between rounds.
        self._noise_rng = random.SystemRandom()

    def protect_update(self, local, global_parameters):
        """Clip the concatenated client delta, then add N(0, (C*sigma)^2) per coordinate.

        This is client update protection, not a record-level epsilon/delta claim.
        Accounting needs an adjacency definition, sampling, rounds and weighting;
        replacement adjacency can require twice the add/remove sensitivity C.
        """
        if self.dp_clip is None:
            return local
        deltas = [p.astype(np.float64) - g for p, g in zip(local, global_parameters)]
        norm = math.sqrt(sum(float(np.square(d).sum()) for d in deltas))
        if not math.isfinite(norm):
            raise ValueError("Non-finite client update; refusing to transmit")
        scale = min(1.0, self.dp_clip / norm) if norm else 1.0
        std = self.dp_clip * self.dp_noise_mult
        protected = []
        for p, g, delta in zip(local, global_parameters, deltas):
            update = delta * scale
            if std:
                noise = np.fromiter((self._noise_rng.gauss(0.0, std) for _ in range(delta.size)),
                                    dtype=np.float64, count=delta.size).reshape(delta.shape)
                update += noise
            value = (g.astype(np.float64) + update).astype(p.dtype)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite protected update; refusing to transmit")
            protected.append(value)
        if not std:
            # Casting back to model precision can round just outside the ball.
            # Move each coordinate toward the reference until the actual wire
            # delta satisfies C too, including bounds below a float32 ULP.
            while math.sqrt(sum(float(np.square(p.astype(np.float64) - g).sum())
                                for p, g in zip(protected, global_parameters))) > self.dp_clip:
                protected = [np.nextafter(p, g) for p, g in zip(protected, global_parameters)]
        return protected

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        reference = [p.copy() for p in parameters]
        self.set_parameters(parameters)
        mu = float(config.get("proximal_mu", 0.0))
        epochs = int(config.get("local_epochs", 2))
        glob = [p.detach().clone() for p in self.model.parameters()]
        t0 = time.perf_counter()
        train_epochs(self.model, self.train_loader, epochs, mu=mu, global_params=glob,
                     optimizer=self.optimizer, loss_fn=self.loss_fn)
        dt = time.perf_counter() - t0
        print(f"FIT mu={mu} epochs={epochs} train_time_s={dt:.2f} peak_rss_mb={peak_rss_mb():.1f}", flush=True)
        protected = self.protect_update(self.get_parameters({}), reference)
        self.set_parameters(protected)
        return protected, len(self.train_loader.dataset), \
            {"train_time_s": dt, "peak_rss_mb": peak_rss_mb(),
             "tls_enabled": self.tls_enabled, "mtls_enabled": self.mtls_enabled,
             # Flower Scalar has no null: zero explicitly denotes disabled clipping.
             "dp_clip": self.dp_clip if self.dp_clip is not None else 0.0,
             "dp_noise_mult": self.dp_noise_mult}

    def evaluate(self, parameters, config):
        """RMSE on held-out *units* of this client (never trained on)."""
        self.set_parameters(parameters)
        r = rmse(self.val_y, predict(self.model, self.val_x))
        return r, len(self.val_y), {"val_rmse": r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-fn", default=os.environ.get("LOSS_FN", "mse"), choices=["mse", "asymmetric"])
    ap.add_argument("--alpha-late", type=float, default=None)
    ap.add_argument("--alpha-early", type=float, default=None)
    ap.add_argument("--ca-cert", default=os.environ.get("FL_CA_CERT"))
    ap.add_argument("--cert", default=os.environ.get("FL_CLIENT_CERT"))
    ap.add_argument("--key", default=os.environ.get("FL_CLIENT_KEY"))
    ap.add_argument("--insecure", action="store_true", help="Local development only: allow plaintext")
    ap.add_argument("--dp-clip", type=float, default=os.environ.get("DP_CLIP"),
                    help="L2 bound C on the full client delta (>0); disabled by default")
    ap.add_argument("--dp-noise-mult", type=float, default=os.environ.get("DP_NOISE_MULT", "0"),
                    help="Gaussian std = C * multiplier (>=0); no epsilon/delta accounting")
    args = ap.parse_args()
    try:
        validate_dp(args.dp_clip, args.dp_noise_mult)
    except ValueError as exc:
        ap.error(str(exc))
    certificates = transport_certificates(ap, args)

    loss_kwargs = {}
    if args.loss_fn == "asymmetric":
        if args.alpha_late is not None:
            loss_kwargs["alpha_late"] = args.alpha_late
        elif "ALPHA_LATE" in os.environ:
            loss_kwargs["alpha_late"] = float(os.environ["ALPHA_LATE"])
        if args.alpha_early is not None:
            loss_kwargs["alpha_early"] = args.alpha_early
        elif "ALPHA_EARLY" in os.environ:
            loss_kwargs["alpha_early"] = float(os.environ["ALPHA_EARLY"])
    loss_fn = get_loss_fn(args.loss_fn, **loss_kwargs)

    seed = int(os.environ.get("SEED", 42))
    cid = os.environ.get("CLIENT_ID")
    if cid is None:
        sys.exit("CLIENT_ID must be set (matches a key in partition.json)")
    fl_server = os.environ.get("FL_SERVER")
    if not fl_server:
        sys.exit("FL_SERVER must be set, e.g. FL_SERVER=<server-ip>:8081")
    part = load_partition()
    if cid not in part:
        sys.exit(f"CLIENT_ID={cid!r} not in partition.json keys {list(part)}")
    set_seed(seed + int(cid))

    df = apply_scale(load_cmapss_raw('train_FD002.txt'), load_scales())
    tr_units, va_units = split_units(part[cid], 0.1, seed=seed)
    X, y = make_xy(df, tr_units)
    Xv, yv = make_xy(df, va_units)
    loader = make_loader(X, y, 64, seed=seed + int(cid))
    print(f"client {cid}: {len(tr_units)} train units ({len(X)} windows), {len(va_units)} val units, loss={args.loss_fn}", flush=True)
    client = SupervisedClient(CmapssRegressor(), loader, torch.tensor(Xv), yv, loss_fn=loss_fn,
                              dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise_mult,
                              tls_enabled=certificates is not None,
                              mtls_enabled=certificates is not None and certificates[1] is not None)
    with flower_client_credentials(certificates):
        fl.client.start_client(server_address=fl_server, client=client.to_client(),
                               root_certificates=certificates[0] if certificates else None,
                               insecure=args.insecure)
