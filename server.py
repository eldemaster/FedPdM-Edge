import argparse
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import ssl
import sys
from types import SimpleNamespace

import flwr as fl
import numpy as np
import torch
from collections import OrderedDict

from common import (CmapssRegressor, set_seed, load_scales, build_test_set, nasa_score, rmse, predict)


def transport_args(argv=None):
    ap = argparse.ArgumentParser(description="Flower server: TLS required unless --insecure is explicit")
    ap.add_argument("--ca-cert", default=os.environ.get("FL_CA_CERT"))
    ap.add_argument("--cert", default=os.environ.get("FL_SERVER_CERT"))
    ap.add_argument("--key", default=os.environ.get("FL_SERVER_KEY"))
    ap.add_argument("--require-client-cert", action=argparse.BooleanOptionalAction, default=True,
                    help="Require a CA-signed client certificate (default: true with TLS)")
    ap.add_argument("--insecure", action="store_true", help="Local development only: allow plaintext")
    args = ap.parse_args(argv)
    paths = (args.ca_cert, args.cert, args.key)
    if args.insecure:
        if any(paths):
            ap.error("--insecure conflicts with certificate options/environment variables")
        print("WARNING: INSECURE plaintext Flower transport; local development only", file=sys.stderr)
        return args, None
    if not all(paths):
        ap.error("TLS requires --ca-cert, --cert and --key (or FL_CA_CERT/FL_SERVER_CERT/FL_SERVER_KEY); "
                 "use --insecure only for local development")
    try:
        # Check PEM parsing and key/certificate matching before starting a listener.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_verify_locations(cafile=args.ca_cert)
        ctx.load_cert_chain(args.cert, args.key)
        certificates = tuple(Path(p).read_bytes() for p in paths)
    except (OSError, ssl.SSLError) as exc:
        ap.error(f"Invalid TLS credentials: {exc}")
    return args, certificates


def flower_server_grpc_module():
    """Resolve and validate the Flower gRPC credential module used by start_server."""
    start_server = inspect.unwrap(fl.server.start_server)
    if "certificates" not in inspect.signature(fl.server.start_server).parameters:
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation")
    start_server_mod = inspect.getmodule(start_server)
    start_grpc_server = getattr(start_server_mod, "start_grpc_server", None)
    try:
        start_server_src = inspect.getsource(start_server)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation") from exc
    if start_grpc_server is None or "start_grpc_server(" not in start_server_src:
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation")

    start_grpc_server = inspect.unwrap(start_grpc_server)
    grpc_server_mod = inspect.getmodule(start_grpc_server)
    factory = getattr(grpc_server_mod, "generic_create_grpc_server", None)
    try:
        start_grpc_src = inspect.getsource(start_grpc_server)
        factory_src = inspect.getsource(inspect.unwrap(factory)) if factory is not None else ""
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation") from exc
    if (factory is None
            or "generic_create_grpc_server(" not in start_grpc_src
            or "grpc.ssl_server_credentials(" not in factory_src):
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation")

    flower_grpc = inspect.getmodule(inspect.unwrap(factory))
    if (flower_grpc is None or not hasattr(flower_grpc, "grpc")
            or not hasattr(flower_grpc.grpc, "ssl_server_credentials")):
        raise RuntimeError(f"Unsupported Flower {fl.__version__} TLS implementation")
    return flower_grpc


@contextmanager
def flower_server_credentials(certificates, require_client_cert):
    """Adapt Flower's gRPC server factory, which hardcodes require_client_auth=False.

    start_server(certificates=...) remains the entry point. Replace only the grpc
    reference in Flower's resolved credential module, for this process's single server.
    Restore it even on failure; never change grpc's global credential factory.
    Refuse an unknown implementation instead of silently running one-way TLS.
    """
    if certificates is None:
        yield
        return
    flower_grpc = flower_server_grpc_module()
    original = flower_grpc.grpc

    def credentials(private_key_certificate_chain_pairs, root_certificates=None,
                    require_client_auth=False):
        ca, cert, key = certificates
        if (private_key_certificate_chain_pairs != ((key, cert),)
                or root_certificates != ca):
            raise RuntimeError("Flower supplied unexpected TLS credentials")
        return original.ssl_server_credentials(
            ((key, cert),), root_certificates=ca, require_client_auth=require_client_cert)

    flower_grpc.grpc = SimpleNamespace(**{**vars(original), "ssl_server_credentials": credentials})
    try:
        yield
    finally:
        flower_grpc.grpc = original


SECURITY_KEYS = ("tls_enabled", "mtls_enabled", "dp_clip", "dp_noise_mult")


def fit_metrics(metrics):
    # Privacy settings must never become sample-weighted averages.
    out = weighted_avg([(n, {k: v for k, v in m.items() if k not in SECURITY_KEYS})
                        for n, m in metrics])
    for key in SECURITY_KEYS:
        values = [m.get(key) for _, m in metrics]
        if values and values[0] is not None and all(v == values[0] for v in values):
            out[key] = values[0]
    return out


class AuditedFedAvg(fl.server.strategy.FedAvg):
    """Retain individual, client-reported privacy settings for every successful fit."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_security = []

    def aggregate_fit(self, server_round, results, failures):
        for proxy, result in results:
            self.client_security.append({"round": server_round, "client_id": proxy.cid,
                                         "num_examples": result.num_examples,
                                         **{k: result.metrics.get(k) for k in SECURITY_KEYS}})
        return super().aggregate_fit(server_round, results, failures)

    def privacy_config(self):
        config = {}
        for key in ("dp_clip", "dp_noise_mult"):
            values = [m[key] for m in self.client_security]
            config[key] = values[0] if values and all(v == values[0] for v in values) else None
        return config


def get_evaluate_fn():
    scales = load_scales()
    X_te, y_te = build_test_set(scales)
    X_te_t = torch.tensor(X_te)
    base = np.full_like(y_te, scales['train_mean_rul'])
    b_rmse, b_nasa = rmse(y_te, base), nasa_score(y_te, base)

    def evaluate(server_round, parameters, config):
        model = CmapssRegressor()
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(sd, strict=True)
        p = predict(model, X_te_t)
        r, n = rmse(y_te, p), nasa_score(y_te, p)
        print(f"Round {server_round} - Model RMSE: {r:.2f}, NASA Score: {n:.2f}", flush=True)
        return r, {"model_rmse": r, "baseline_rmse": float(b_rmse), "nasa_score": float(n)}
    return evaluate


def weighted_avg(metrics):
    tot = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys() if metrics else []
    return {k: sum(n * m[k] for n, m in metrics) / tot for k in keys}


if __name__ == "__main__":
    args, certificates = transport_args()
    seed = int(os.environ.get("SEED", 777))
    rounds = int(os.environ.get("ROUNDS", 5))
    n_clients = int(os.environ.get("MIN_CLIENTS", 2))
    mu = float(os.environ.get("PROXIMAL_MU", 0.0))          # >0 => FedProx local objective
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", 2))
    out_dir = os.environ.get("RESULTS_DIR", "results/fd002")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    # Seeded initial weights from the server: otherwise Flower asks a random client,
    # making round 0 depend on which client connects first.
    init = fl.common.ndarrays_to_parameters([v.numpy() for v in CmapssRegressor().state_dict().values()])
    strategy = AuditedFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=n_clients, min_evaluate_clients=n_clients, min_available_clients=n_clients,
        initial_parameters=init,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda r: {"proximal_mu": mu, "local_epochs": local_epochs},
        fit_metrics_aggregation_fn=fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )
    with flower_server_credentials(certificates, args.require_client_cert):
        history = fl.server.start_server(server_address=os.environ.get("BIND", "0.0.0.0:8081"),
                                        config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy,
                                        certificates=certificates)
    if not history.metrics_distributed_fit:
        sys.exit("No fit metrics recorded - clients never completed a round")
    tag = "fedprox" if mu > 0 else "fedavg"
    name = f"results_physical_seed{seed}.json" if mu == 0 else f"results_physical_{tag}_mu{mu}_seed{seed}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump({"metrics_centralized": history.metrics_centralized,
                   "metrics_distributed_fit": history.metrics_distributed_fit,
                   "metrics_distributed": history.metrics_distributed,
                   "client_security": strategy.client_security,
                   "config": {"seed": seed, "rounds": rounds, "mu": mu, "local_epochs": local_epochs,
                              "tls_enabled": certificates is not None,
                              "mtls_enabled": certificates is not None and args.require_client_cert,
                              **strategy.privacy_config()}}, f)
