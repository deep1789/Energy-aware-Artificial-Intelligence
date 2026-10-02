"""Main comparison + ablations + sensitivity using the tuned configurations (eval seeds 0..4)."""
import json, sys
from experiments.common import run_all

BASE_METHODS = ["fedavg", "fedprox", "fedavg_q8", "fedavg_paced", "static_depth", "oort",
                "energy_greedy", "ecofed"]
ABLATIONS = ["ecofed_noQ", "ecofed_noZ", "ecofed_fixdepth", "ecofed_fixprec", "ecofed_fixtau"]
SEEDS = range(8)
MAIN_SEEDS = range(16)
DATASETS = ("uci_har", "pamap2", "speech")
EXP = "m3"


def fold_of(ds, s):
    return s % 4 if ds == "pamap2" else 0


def tuned(ds, phi):
    return json.load(open(f"results/best4_{ds}_{phi}.json"))


def main_jobs(ds, phi, T=80):
    best = tuned(ds, phi)
    jobs = []
    for m in BASE_METHODS:
        for s in MAIN_SEEDS:
            jobs.append(dict(exp=EXP, dataset=ds, fold=fold_of(ds, s), method=m, kw=best[m]["kw"],
                             cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"])))
    for s in MAIN_SEEDS:      # single-exit FedAvg for the lifecycle comparison (tuned FedAvg settings)
        jobs.append(dict(exp=EXP, dataset=ds, fold=fold_of(ds, s), method="fedavg_single", kw={},
                         cfg=dict(phi=phi, T=T, seed=s, lr=best["fedavg"]["lr"])))
    if phi in (0.1, 0.25):
        for m in ABLATIONS:
            for s in SEEDS:
                jobs.append(dict(exp=EXP, dataset=ds, fold=fold_of(ds, s), method=m, kw=best["ecofed"]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best["ecofed"]["lr"])))
    return jobs


def sens_jobs(ds="pamap2", phi=0.25, T=80):
    """Energy-model misspecification and radio-cost regimes, for the tuned methods of the main grid."""
    best = tuned(ds, phi)
    jobs = []
    for radio in (1.0, 10.0, 100.0):
        for m in ("fedavg", "fedavg_q8", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="s3_radio", dataset=ds, fold=fold_of(ds, s), method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], radio_scale=radio)))
    for sigma in (0.25, 0.5):
        for m in ("fedavg_paced", "static_depth", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="s3_err", dataset=ds, fold=fold_of(ds, s), method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], energy_err=sigma)))
    return jobs


def lambda_sweep_jobs(T=80, lams=(0.03, 0.1, 0.3, 1.0), seeds=range(8)):
    """Full sweep of EcoFed's shadow price on the evaluation seeds (reported in full, never used for selection)."""
    jobs = []
    for ds in DATASETS:
        for phi in (0.25, 1.0):
            best = tuned(ds, phi)["ecofed"]
            for lam in lams:
                kw = dict(best["kw"]); kw["q_floor"] = lam
                for s in seeds:
                    jobs.append(dict(exp="s3_lam", dataset=ds, fold=fold_of(ds, s), method="ecofed",
                                     kw=kw, cfg=dict(phi=phi, T=T, seed=s, lr=best["lr"])))
    return jobs


def overhead_jobs(T=80):
    """True per-sample overhead (fitted from CPU latency) that the controllers' nominal energy model ignores."""
    jobs = []
    for ds, ov_list in (("pamap2", (0.5e6, 1.0e6)), ("uci_har", (0.5e6, 1.0e6)), ("speech", (0.5e6, 1.0e6))):
        best = tuned(ds, 0.25)
        for ov in ov_list:
            for m in ("fedavg", "oort", "energy_greedy", "static_depth", "fedavg_paced", "ecofed"):
                for s in SEEDS:
                    jobs.append(dict(exp="s3_ovh", dataset=ds, fold=fold_of(ds, s), method=m,
                                     kw=best[m]["kw"], cfg=dict(phi=0.25, T=T, seed=s, lr=best[m]["lr"], overhead_macs=ov)))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "main":
        for ds in DATASETS:
            for phi in (0.1, 0.25, 1.0):
                jobs += main_jobs(ds, phi)
    elif which == "lam":
        jobs = lambda_sweep_jobs()
    elif which == "ovh":
        jobs = overhead_jobs()
    elif which == "sens":
        jobs = sens_jobs()
    print(len(jobs), "jobs", flush=True)
    run_all(jobs)
    print("done")
