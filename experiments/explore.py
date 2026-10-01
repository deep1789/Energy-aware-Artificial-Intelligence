"""Quick exploratory comparison on tuning seeds (100-102), validation metrics only."""
import json, sys
import numpy as np
from analysis.metrics import acc_at_energy, auc_energy
from experiments.common import run_all

def main(ds, phi, T, methods, seeds=(100, 101, 102), radio=1.0, tag=""):
    jobs = [dict(exp=f"explore_{ds}_{phi}_{radio}{tag}", dataset=ds, fold=0, method=m, kw=kw,
                 cfg=dict(phi=phi, T=T, seed=s, radio_scale=radio), record_final=False)
            for m, kw in methods for s in seeds]
    rows = {}
    for job, res in run_all(jobs):
        rows.setdefault((job["method"], json.dumps(job["kw"])), []).append(res)
    print("method                              val@.02 .05  .1  .25  .5  1.0 |  auc | E/cap alive depth", flush=True)
    for (m, kw), rs in rows.items():
        f = lambda fr: np.mean([acc_at_energy(r, fr, "val_acc", "best") for r in rs])
        auc = np.mean([auc_energy(r, "val_acc", "best") for r in rs])
        ec = np.mean([r["total_energy"] / r["fleet_capacity"] for r in rs])
        al = np.mean([r["log"]["alive"][-1] for r in rs])
        dp = np.mean([np.mean(r["log"]["depth_mean"]) for r in rs])
        print(f"{m + kw:36s} {f(.02):.3f} {f(.05):.3f} {f(.1):.3f} {f(.25):.3f} {f(.5):.3f} {f(1):.3f} | {auc:.3f} | {ec:.2f} {al:.1f} {dp:.2f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4]),
         radio=float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
