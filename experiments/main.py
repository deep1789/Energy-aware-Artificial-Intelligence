"""Main comparison + ablations + sensitivity using the tuned configurations (eval seeds 0..4)."""
import json, sys
from experiments.common import run_all

BASE_METHODS = ["fedavg", "fedprox", "fedavg_q8", "fedavg_paced", "static_depth", "oort",
                "energy_greedy", "ecofed"]
ABLATIONS = ["ecofed_noQ", "ecofed_noZ", "ecofed_fixdepth", "ecofed_fixprec", "ecofed_fixtau"]
SEEDS = range(5)


def tuned(ds, phi):
    return json.load(open(f"results/best_{ds}_{phi}.json"))


def main_jobs(ds, phi, T=80):
    best = tuned(ds, phi)
    jobs = []
    for m in BASE_METHODS:
        for s in SEEDS:
            jobs.append(dict(exp="main", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=m,
                             kw=best[m]["kw"], cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"])))
    if phi in (0.1, 0.25):
        for m in ABLATIONS:
            for s in SEEDS:
                jobs.append(dict(exp="main", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=m,
                                 kw=best["ecofed"]["kw"], cfg=dict(phi=phi, T=T, seed=s, lr=best["ecofed"]["lr"])))
    return jobs


def sens_jobs(ds="uci_har", phi=0.25, T=80):
    """Energy-model misspecification and radio-cost regimes, for the tuned methods of the main grid."""
    best = tuned(ds, phi)
    jobs = []
    for radio in (1.0, 10.0, 100.0):
        for m in ("fedavg", "fedavg_q8", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="sens_radio", dataset=ds, fold=0, method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], radio_scale=radio)))
    for sigma in (0.25, 0.5):
        for m in ("fedavg_paced", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="sens_err", dataset=ds, fold=0, method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], energy_err=sigma)))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "main":
        for ds in ("uci_har", "pamap2"):
            for phi in (0.1, 0.25, 1.0):
                jobs += main_jobs(ds, phi)
    elif which == "sens":
        jobs = sens_jobs()
    print(len(jobs), "jobs", flush=True)
    run_all(jobs)
    print("done")
