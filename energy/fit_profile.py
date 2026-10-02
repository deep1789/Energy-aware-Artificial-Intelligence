"""Fit a DeviceProfile (alpha J/MAC, beta J/byte, fixed per-sample overhead) from metered measurements.

Input CSV (one row per measured configuration), produced by energy/measure_hw.py or by hand from any power meter:
    depth, batch, samples, joules_above_idle
Model per sample:  E = alpha * 3 * MAC(depth) + beta * Bytes(depth, batch) + c0     (alpha, beta, c0 >= 0, NNLS)
c0 is the fixed per-sample overhead that Section 6.9 of the paper shows exists on a real CPU. Radio energies come from
a separate transfer measurement (bytes sent/received and joules above idle):  eps = joules / (8 * bytes).
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.optimize import nnls

from energy.model import BATCH, BYTES_PER_PARAM, EnergyModel
from models.multiexit import MultiExitCNN1D


def design(rows, in_ch, n_classes, in_len=128):
    m = MultiExitCNN1D(in_ch, n_classes, in_len)
    em = EnergyModel(m.block_stats(), in_ch, in_len)
    A, y = [], []
    for r in rows:
        d, b, n, j = int(r["depth"]), int(r["batch"]), float(r["samples"]), float(r["joules_above_idle"])
        act = em.in_elems + sum(s.act_elems for s in em.stats[:d])
        weights = sum(s.params for s in em.stats[:d])
        bytes_ps = BYTES_PER_PARAM * (4 * act + 4 * weights / b)
        A.append([3.0 * em.macs(d), bytes_ps, 1.0])
        y.append(j / n)
    return np.array(A), np.array(y)


def fit(rows, in_ch=9, n_classes=6):
    A, y = design(rows, in_ch, n_classes)
    scale = A.max(axis=0)
    coef, resid = nnls(A / scale, y)
    coef = coef / scale
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float(np.mean(np.abs(pred - y) / y))
    return dict(alpha=float(coef[0]), beta=float(coef[1]), overhead_j_per_sample=float(coef[2]), r2=float(r2), mape=mape)


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    rows = read_rows(sys.argv[1])
    out = fit(rows, int(sys.argv[2]) if len(sys.argv) > 2 else 9, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    print(json.dumps(out, indent=1))
