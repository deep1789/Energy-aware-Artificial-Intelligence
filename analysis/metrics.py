"""Metrics computed from run logs. Energy is expressed as a fraction of fleet capacity (sum of batteries)."""
from __future__ import annotations

import numpy as np


def curve(res, key="acc", exit_idx=-1):
    """(energy_fraction, accuracy) arrays with the origin at chance level omitted."""
    log = res["log"]
    x = np.array(log["cum_energy"]) / res["fleet_capacity"]
    y = np.array([a[exit_idx] for a in log[key]])
    return x, y


def acc_at_energy(res, frac, key="acc", exit_idx=-1, smooth=3):
    """Accuracy (mean of the last `smooth` evals) at the first eval where cumulative energy >= frac.
    If the fleet never spends `frac` of capacity, returns the final value (battery exhausted)."""
    x, y = curve(res, key, exit_idx)
    i = int(np.searchsorted(x, frac))
    i = min(i, len(y) - 1)
    return float(np.mean(y[max(0, i - smooth + 1):i + 1]))


def auc_energy(res, key="acc", exit_idx=-1, upto=1.0):
    """Area under accuracy vs energy-fraction on [0, upto]; accuracy is held at its last value."""
    x, y = curve(res, key, exit_idx)
    grid = np.linspace(0, upto, 101)
    return float(np.trapezoid(np.interp(grid, x, y, left=y[0]), grid) / upto)


def energy_to_target(res, target, key="acc", exit_idx=-1, smooth=3):
    """Energy fraction at which smoothed accuracy first reaches `target`; nan if never."""
    x, y = curve(res, key, exit_idx)
    ys = np.array([np.mean(y[max(0, i - smooth + 1):i + 1]) for i in range(len(y))])
    hit = np.where(ys >= target)[0]
    return float(x[hit[0]]) if len(hit) else float("nan")


def final_acc(res, key="acc", exit_idx=-1, last=3):
    return float(np.mean([a[exit_idx] for a in res["log"][key][-last:]]))


def jain(v):
    v = np.asarray(v, float)
    return float(v.sum() ** 2 / (len(v) * (v ** 2).sum() + 1e-12))


def lifetime(res, frac=0.1):
    """Round at which the number of live clients first drops below (1-frac)*N; T if never."""
    log = res["log"]
    N = max(log["alive"][0], 1)
    N = len(res["per_client_energy"])
    for r, a in zip(log["round"], log["alive"]):
        if a <= (1 - frac) * N:
            return r
    return log["round"][-1]
