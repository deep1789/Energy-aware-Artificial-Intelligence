"""Hyper-parameter tuning on dedicated seeds (100,101) with validation accuracy only; <= 6 trials per method."""
import json, sys
from pathlib import Path
import numpy as np
from analysis.metrics import auc_energy
from experiments.common import run_all

GRIDS = {
    "fedprox": [{"mu": m} for m in (0.001, 0.01, 0.1)],
    "oort": [{"eps": e, "xi": x} for e in (0.1, 0.3) for x in (1.0, 2.0)],
    "energy_greedy": [{"eps": e} for e in (0.1, 0.3)],
    "ecofed": [{"V": v, "rho_min": r} for v in (1.0, 2.0, 4.0) for r in (0.3, 0.6)],
    "fedavg": [{}], "fedavg_q8": [{}], "fedavg_paced": [{}], "static_depth": [{}],
}

if __name__ == "__main__":
    ds = sys.argv[1]; phi = float(sys.argv[2]); T = int(sys.argv[3])
    jobs = []
    for meth, grid in GRIDS.items():
        for kw in grid:
            for seed in (100, 101):
                jobs.append(dict(exp=f"tune_{ds}", dataset=ds, fold=0, method=meth, kw=kw,
                                 cfg=dict(phi=phi, T=T, seed=seed), record_final=False))
    out = {}
    for job, res in run_all(jobs):
        out.setdefault((job["method"], json.dumps(job["kw"], sort_keys=True)), []).append(
            auc_energy(res, "val_acc"))
    best = {}
    for (m, kw), v in sorted(out.items()):
        print(f"{m:15s} {kw:40s} val-AUC={np.mean(v):.4f}")
        if m not in best or np.mean(v) > best[m][0]:
            best[m] = (float(np.mean(v)), json.loads(kw))
    Path("results").mkdir(exist_ok=True)
    json.dump({m: kw for m, (_, kw) in best.items()}, open(f"results/best_{ds}.json", "w"), indent=1)
    print(json.dumps({m: kw for m, (_, kw) in best.items()}))
