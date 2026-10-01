"""Lifecycle analysis: confidence-gated early-exit inference energy and training/inference energy crossover.

For each trained model (final softmax outputs on validation and test) we pick, on VALIDATION data only, the
single confidence threshold theta that minimises expected inference energy subject to cascade accuracy >=
(best single-exit validation accuracy - delta). The test accuracy / energy of that cascade are then reported.
Inference energy per exit uses the fleet-average device (mix weights from energy.model.DEVICE_MIX)."""
from __future__ import annotations

import numpy as np

from energy.model import DEVICE_MIX, DEVICE_PROFILES, EnergyModel
from models.multiexit import MultiExitCNN1D

THETAS = np.concatenate([np.linspace(0.3, 0.99, 36), [0.995, 0.999, 1.01]])


def exit_energies(in_ch, n_classes, in_len=128):
    m = MultiExitCNN1D(in_ch, n_classes, in_len)
    em = EnergyModel(m.block_stats(), in_ch, in_len)
    return np.array([sum(f * em.e_infer_exit(DEVICE_PROFILES[n], j) for n, f in DEVICE_MIX)
                     for j in range(1, em.L + 1)])


def cascade(probs, y, theta, e_exit, allowed_exits=None):
    """probs (n, L, C). Exit at the first allowed exit whose max-prob >= theta; the last allowed exit is forced."""
    n, L, _ = probs.shape
    allowed = list(range(L)) if allowed_exits is None else list(allowed_exits)
    conf = probs.max(-1)
    pred = probs.argmax(-1)
    chosen = np.full(n, allowed[-1])
    done = np.zeros(n, bool)
    for j in allowed[:-1]:
        hit = (~done) & (conf[:, j] >= theta)
        chosen[hit] = j
        done |= hit
    acc = float((pred[np.arange(n), chosen] == y).mean())
    energy = float(e_exit[chosen].mean())
    exit_frac = np.bincount(chosen, minlength=L) / n
    return acc, energy, exit_frac


def calibrate_and_test(val_probs, y_val, test_probs, y_test, e_exit, delta=0.01, allowed_exits=None):
    val_probs, test_probs = val_probs.astype(np.float32), test_probs.astype(np.float32)
    L = val_probs.shape[1]
    single_val = [(val_probs[:, j].argmax(-1) == y_val).mean() for j in range(L)]
    jb = int(np.argmax(single_val))
    target = single_val[jb] - delta
    best = None
    for th in THETAS:
        a, e, _ = cascade(val_probs, y_val, th, e_exit, allowed_exits)
        if a >= target and (best is None or e < best[1]):
            best = (th, e)
    if best is None:
        best = (THETAS[-1], float(e_exit[jb]))
    th = best[0]
    acc, en, frac = cascade(test_probs, y_test, th, e_exit, allowed_exits)
    single_test_acc = float((test_probs[:, jb].argmax(-1) == y_test).mean())
    return dict(theta=float(th), cascade_acc=acc, cascade_energy=en, exit_frac=frac,
                single_exit=jb, single_acc=single_test_acc, single_energy=float(e_exit[jb]))
