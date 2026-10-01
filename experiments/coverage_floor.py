"""Probe of Theorem 1: final full-objective gradient norm vs depth-coverage rho (no energy constraint)."""
import sys
import numpy as np
from experiments.common import run_all

RHOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

if __name__ == "__main__":
    ds = sys.argv[1]; T = int(sys.argv[2]); seeds = range(int(sys.argv[3]))
    jobs = [dict(exp=f"cov_{ds}", dataset=ds, fold=0, method="forced_depth", kw={"rho": r},
                 cfg=dict(phi=1000.0, T=T, seed=s, eval_every=T, dense_until=0)) for r in RHOS for s in seeds]
    run_all(jobs)
    print("done", len(jobs))
