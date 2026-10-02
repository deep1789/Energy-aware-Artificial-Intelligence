"""Equal-budget tuning: every method gets exactly N_TRIALS random-search trials (2 tuning seeds each), drawn without
replacement from its own space = (method knobs) x (16 learning rates), scored by validation AUC only."""
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from analysis.metrics import auc_energy
from experiments.common import run_all

N_TRIALS = 14
SEEDS = (100, 101)
LRS = [round(float(x), 3) for x in np.geomspace(0.02, 0.8, 16)]
KNOBS = {
    "fedavg": [{}], "fedavg_q8": [{}], "fedavg_paced": [{}], "static_depth": [{}],
    "fedprox": [{"mu": m} for m in (0.001, 0.01, 0.1)],
    "oort": [{"eps": e, "xi": x} for e in (0.1, 0.3) for x in (1.0, 2.0)],
    "energy_greedy": [{"eps": e} for e in (0.1, 0.3)],
    "ecofed": [{"V": 2.0, "rho_min": r, "q_floor": q, "gamma_pow": g}
               for r in (0.3, 0.6) for q in (0.03, 0.1, 0.3) for g in (1.0, 0.5, 0.25)],
}


def trials(ds, phi, method):
    space = [(kw, lr) for kw in KNOBS[method] for lr in LRS]
    h = int(hashlib.md5(f"{ds}|{phi}|{method}".encode()).hexdigest()[:8], 16)
    idx = np.random.default_rng(h).permutation(len(space))[:N_TRIALS]
    return [space[i] for i in idx]


def fold_of(ds, s):
    return s % 4 if ds == "pamap2" else 0


if __name__ == "__main__":
    ds, phi, T = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    jobs = [dict(exp=f"tune4_{ds}_{phi}", dataset=ds, fold=fold_of(ds, s), method=m, kw=kw,
                 cfg=dict(phi=phi, T=T, seed=s, lr=lr), record_final=False)
            for m in KNOBS for kw, lr in trials(ds, phi, m) for s in SEEDS]
    groups = {}
    for job, r in run_all(jobs):
        groups.setdefault((job["method"], json.dumps(job["kw"], sort_keys=True), job["cfg"]["lr"]), []).append(auc_energy(r, "val_acc", "best"))
    best = {}
    for (m, kw, lr), v in groups.items():
        sc = float(np.mean(v))
        if m not in best or sc > best[m]["val_auc"]:
            best[m] = dict(kw=json.loads(kw), lr=lr, val_auc=sc, n_trials=N_TRIALS)
    Path("results").mkdir(exist_ok=True)
    json.dump(best, open(f"results/best4_{ds}_{phi}.json", "w"), indent=1)
    for m, d in best.items():
        print(f"{ds} phi={phi} {m:15s} auc={d['val_auc']:.4f} lr={d['lr']} {d['kw']}")
