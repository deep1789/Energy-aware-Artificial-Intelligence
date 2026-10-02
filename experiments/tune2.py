"""Two-stage tuning on dedicated seeds (100,101) using validation accuracy only (<= 14 trials per method).

Stage A: method-specific knobs at learning rate 0.12. Stage B: learning rate {0.06, 0.25} around the
best Stage-A configuration. Criterion: area under best-exit validation accuracy vs energy fraction."""
import json, sys
from pathlib import Path
import numpy as np
from analysis.metrics import auc_energy
from experiments.common import run_all

STAGE_A = {
    "fedavg": [{}], "fedavg_q8": [{}], "fedavg_paced": [{}], "static_depth": [{}],
    "fedprox": [{"mu": m} for m in (0.001, 0.01, 0.1)],
    "oort": [{"eps": e, "xi": x} for e in (0.1, 0.3) for x in (1.0, 2.0)],
    "energy_greedy": [{"eps": e} for e in (0.1, 0.3)],
    "ecofed": [{"V": 2.0, "rho_min": r, "q_floor": q, "gamma_pow": g}
               for r in (0.3, 0.6) for q in (0.03, 0.1, 0.3) for g in (1.0, 0.5)],
}
SEEDS = (100, 101)


def evaluate(ds, phi, T, method, kw, lr):
    jobs = [dict(exp=f"tune3_{ds}_{phi}", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=method, kw=kw,
                 cfg=dict(phi=phi, T=T, seed=s, lr=lr), record_final=False) for s in SEEDS]
    return jobs


def score(results):
    return float(np.mean([auc_energy(r, "val_acc", "best") for _, r in results]))


if __name__ == "__main__":
    ds, phi, T = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    jobs = [j for m, g in STAGE_A.items() for kw in g for j in evaluate(ds, phi, T, m, kw, 0.12)]
    res = run_all(jobs)
    groups = {}
    for job, r in res:
        groups.setdefault((job["method"], json.dumps(job["kw"], sort_keys=True)), []).append((job, r))
    best = {}
    for (m, kw), v in groups.items():
        sc = score(v)
        if m not in best or sc > best[m][0]:
            best[m] = (sc, json.loads(kw), 0.12)
    jobsB = [j for m, (_, kw, _) in best.items() for lr in (0.06, 0.25) for j in evaluate(ds, phi, T, m, kw, lr)]
    resB = run_all(jobsB)
    gB = {}
    for job, r in resB:
        gB.setdefault((job["method"], job["cfg"]["lr"]), []).append((job, r))
    for (m, lr), v in gB.items():
        sc = score(v)
        if sc > best[m][0]:
            best[m] = (sc, best[m][1], lr)
    out = {m: dict(kw=kw, lr=lr, val_auc=sc) for m, (sc, kw, lr) in best.items()}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(f"results/best3_{ds}_{phi}.json", "w"), indent=1)
    for m, d in out.items():
        print(f"{ds} phi={phi} {m:15s} auc={d['val_auc']:.4f} lr={d['lr']} {d['kw']}")
