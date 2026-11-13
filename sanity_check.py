import numpy as np
from common import (CmapssRegressor, USEFUL_SENSORS, set_seed, load_cmapss_raw, load_scales, apply_scale,
                    build_test_set, make_xy, make_loader, train_epochs, predict, rmse)


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
    print("Sanity check passed.")
