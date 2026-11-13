import numpy as np
import torch
from common import (CmapssRegressor, USEFUL_SENSORS, AsymmetricRULLoss, get_loss_fn,
                    set_seed, load_cmapss_raw, load_scales, apply_scale,
                    build_test_set, make_xy, make_loader, train_epochs, predict, rmse)


def run_loss_sanity_check():
    """Verify AsymmetricRULLoss asymmetry, finite gradients, and CmapssRegressor integration."""
    loss_fn = get_loss_fn("asymmetric")
    assert isinstance(loss_fn, AsymmetricRULLoss)
    y_true = torch.tensor([50.0, 50.0])
    y_late = torch.tensor([60.0, 60.0], requires_grad=True)   # +10 (overestimation / late)
    y_early = torch.tensor([40.0, 40.0], requires_grad=True)  # -10 (underestimation / early)
    l_late = loss_fn(y_late, y_true)
    l_early = loss_fn(y_early, y_true)
    assert l_late.item() > l_early.item(), (
        f"Expected late penalty ({l_late.item()}) > early penalty ({l_early.item()})"
    )
    l_late.backward()
    l_early.backward()
    assert torch.isfinite(y_late.grad).all() and torch.isfinite(y_early.grad).all()

    # Integration with CmapssRegressor
    set_seed(42)
    model = CmapssRegressor()
    bx = torch.randn(8, 30, len(USEFUL_SENSORS))
    by = torch.rand(8) * 100.0
    out = model(bx)
    loss = loss_fn(out, by)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return float(l_late.item()), float(l_early.item())


def run_sanity_check(seed=42):
    set_seed(seed)
    scales = load_scales()
    X_te, y_te = build_test_set(scales)
    baseline = rmse(y_te, np.full_like(y_te, scales['train_mean_rul']))
    model = CmapssRegressor()
    untrained = rmse(y_te, predict(model, X_te))
    df = apply_scale(load_cmapss_raw('train_FD002.txt'), scales)
    X, y = make_xy(df, df['unit'].unique())
    train_epochs(model, make_loader(X, y, 64, seed=seed), epochs=2)
    trained = rmse(y_te, predict(model, X_te))
    return len(y_te), baseline, untrained, trained


if __name__ == "__main__":
    n, baseline, untrained, trained = run_sanity_check(42)
    print(f"Sanity Check (n_test={n}) -> Baseline RMSE: {baseline:.2f}, Untrained RMSE: {untrained:.2f}, Quick-Trained RMSE: {trained:.2f}")
    assert n == 259, f"FD002 test set must contain all 259 engines, got {n}"
    assert untrained > baseline, "untrained net must be worse than a constant"
    assert trained < baseline * 0.75, "model must beat the constant baseline by 25%"
    l_late, l_early = run_loss_sanity_check()
    print(f"AsymmetricRULLoss Check -> Late(+10): {l_late:.2f} > Early(-10): {l_early:.2f}, gradients finite.")
    print("Sanity check passed.")
