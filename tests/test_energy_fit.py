import numpy as np

from energy import fit_profile as FP


def synth_rows(alpha, beta, c0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in (1, 2, 3, 4):
        for b in (16, 32, 64):
            rows.append(dict(depth=d, batch=b, samples=1000.0, joules_above_idle=0.0))
    A, _ = FP.design(rows, 9, 6)
    for r, a in zip(rows, A):
        r["joules_above_idle"] = 1000.0 * (a @ np.array([alpha, beta, c0])) * (1 + noise * rng.normal())
    return rows


def test_fit_recovers_noise_free_parameters():
    # scale-aware recovery of parameters from an exactly generated dataset
    rows = synth_rows(7e-10, 4e-10, 3e-4)
    out = FP.fit(rows)
    assert out["r2"] > 0.999 and out["mape"] < 1e-3
    assert abs(out["alpha"] - 7e-10) / 7e-10 < 0.05
    assert abs(out["overhead_j_per_sample"] - 3e-4) / 3e-4 < 0.05


def test_fit_is_robust_to_5pct_noise():
    out = FP.fit(synth_rows(7e-10, 4e-10, 3e-4, noise=0.05, seed=3))
    assert out["r2"] > 0.95 and out["alpha"] > 0
