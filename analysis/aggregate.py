"""Aggregate main-experiment runs into a per-run metric table plus summary statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.metrics import acc_at_energy, auc_energy, final_acc, jain
from experiments.common import load_results

FRACS = (0.1, 0.25, 0.5, 1.0)


def eta_abs(res, target, smooth=3):
    """Absolute fleet energy (J) at which smoothed best-exit test accuracy first reaches target (linear interp)."""
    x = np.array(res["log"]["cum_energy"])
    va = np.array(res["log"]["val_acc"])
    y = np.array([a[int(np.argmax(v))] for a, v in zip(res["log"]["acc"], va)])
    ys = np.array([y[max(0, i - smooth + 1):i + 1].mean() for i in range(len(y))])
    hit = np.where(ys >= target)[0]
    if not len(hit):
        return np.nan
    i = hit[0]
    if i == 0:
        return float(x[0])
    f = (target - ys[i - 1]) / max(ys[i] - ys[i - 1], 1e-9)
    return float(x[i - 1] + f * (x[i] - x[i - 1]))


def checkpoint(res, smooth=3):
    """Validation-selected checkpoint: the evaluation point with the highest smoothed best-exit validation accuracy.
    Returns the test accuracy of that checkpoint and the fleet energy spent until it."""
    log = res["log"]
    va = np.array(log["val_acc"])
    vbest = va.max(axis=1)
    vs = np.array([vbest[max(0, i - smooth + 1):i + 1].mean() for i in range(len(vbest))])
    i = int(np.argmax(vs))
    acc = log["acc"][i][int(np.argmax(va[i]))]
    return dict(ckpt_acc=float(acc), ckpt_E=float(log["cum_energy"][i]), ckpt_round=int(log["round"][i]))


def per_run_table(exp="main"):
    rows = []
    for job, r in load_results(exp):
        cfg = job["cfg"]
        cap = r["fleet_capacity"]
        row = dict(dataset=job["dataset"], phi=cfg["phi"], method=job["method"], seed=cfg["seed"],
                   radio=cfg.get("radio_scale", 1.0), err=cfg.get("energy_err", 0.0),
                   total_E=r["total_energy"], cap=cap, auc=auc_energy(r, "acc", "best"),
                   final_best=final_acc(r, "acc", "best"), final_exit4=final_acc(r, "acc", -1),
                   final_exit1=final_acc(r, "acc", 0), secs=r["secs"],
                   jain=jain(np.array(r["per_client_energy"]) / np.array(r["per_client_capacity"])),
                   depleted=r["log"].get("depleted", [np.nan])[-1] / len(r["per_client_energy"]),
                   rho2=np.mean([x[1] for x in r["log"]["rho"]]), rho4=np.mean([x[3] for x in r["log"]["rho"]]),
                   depth=np.mean(r["log"]["depth_mean"]), gradnorm=r.get("gradnorm_sq", np.nan))
        for f in FRACS:
            row[f"acc{int(f*100)}"] = acc_at_energy(r, f, "acc", "best")
        row.update(checkpoint(r))
        row["_res"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def add_eta(df, ref_method="fedavg", ref_phi=1.0, fracs=(0.9, 0.95)):
    """ETA targets are fractions of the FedAvg plateau at the loosest budget (per dataset)."""
    ref = (df[(df.method == ref_method) & (df.phi == ref_phi)].groupby("dataset").final_best.mean())
    for fr in fracs:
        df[f"eta{int(fr*100)}"] = [eta_abs(r, fr * ref[d]) for r, d in zip(df["_res"], df["dataset"])]
    df.attrs["ref_plateau"] = ref.to_dict()
    return df


def paired_test(df, metric, a="ecofed", b="fedavg", lower_better=False):
    """Paired Wilcoxon signed-rank across seeds (seed = pairing key) + mean difference + bootstrap CI."""
    A = df[df.method == a].set_index("seed")[metric]
    B = df[df.method == b].set_index("seed")[metric]
    idx = A.index.intersection(B.index)
    d = (A.loc[idx] - B.loc[idx]).dropna().values
    if len(d) < 3:
        return dict(n=len(d), diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    try:
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(n=len(d), diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), p=float(p))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out
