"""Main comparison + ablations + sensitivity using the tuned configurations (eval seeds 0..4)."""
import json, sys
from experiments.common import run_all

BASE_METHODS = ["fedavg", "fedprox", "fedavg_q8", "fedavg_paced", "static_depth", "oort",
                "energy_greedy", "ecofed"]
ABLATIONS = ["ecofed_noQ", "ecofed_noZ", "ecofed_fixdepth", "ecofed_fixprec", "ecofed_fixtau"]
SEEDS = range(8)


def tuned(ds, phi):
    return json.load(open(f"results/best3_{ds}_{phi}.json"))


def main_jobs(ds, phi, T=80):
    best = tuned(ds, phi)
    jobs = []
    for m in BASE_METHODS:
        for s in SEEDS:
            jobs.append(dict(exp="m2", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=m,
                             kw=best[m]["kw"], cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"])))
    for s in SEEDS:      # single-exit FedAvg for the lifecycle comparison (tuned FedAvg settings)
        jobs.append(dict(exp="m2", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method="fedavg_single",
                         kw={}, cfg=dict(phi=phi, T=T, seed=s, lr=best["fedavg"]["lr"])))
    if phi in (0.1, 0.25):
        for m in ABLATIONS:
            for s in SEEDS:
                jobs.append(dict(exp="m2", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=m,
                                 kw=best["ecofed"]["kw"], cfg=dict(phi=phi, T=T, seed=s, lr=best["ecofed"]["lr"])))
    return jobs


def extra_seed_jobs(seeds=range(8, 16), T=80):
    """Additional evaluation seeds for the main comparison (no ablations)."""
    jobs = []
    for ds in ("uci_har", "pamap2"):
        for phi in (0.1, 0.25, 1.0):
            best = tuned(ds, phi)
            for m in BASE_METHODS + ["fedavg_single"]:
                for s in seeds:
                    kw = {} if m == "fedavg_single" else best[m]["kw"]
                    lr = best["fedavg" if m == "fedavg_single" else m]["lr"]
                    jobs.append(dict(exp="m2", dataset=ds, fold=0 if ds == "uci_har" else s % 4, method=m, kw=kw,
                                     cfg=dict(phi=phi, T=T, seed=s, lr=lr)))
    return jobs


def sens_jobs(ds="pamap2", phi=0.25, T=80):
    """Energy-model misspecification and radio-cost regimes, for the tuned methods of the main grid."""
    best = tuned(ds, phi)
    jobs = []
    for radio in (1.0, 10.0, 100.0):
        for m in ("fedavg", "fedavg_q8", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="sens_radio", dataset=ds, fold=s % 4 if ds == "pamap2" else 0, method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], radio_scale=radio)))
    for sigma in (0.25, 0.5):
        for m in ("fedavg_paced", "static_depth", "oort", "energy_greedy", "ecofed"):
            for s in SEEDS:
                jobs.append(dict(exp="sens_err", dataset=ds, fold=s % 4 if ds == "pamap2" else 0, method=m, kw=best[m]["kw"],
                                 cfg=dict(phi=phi, T=T, seed=s, lr=best[m]["lr"], energy_err=sigma)))
    return jobs


def lambda_sweep_jobs(T=80, lams=(0.03, 0.1, 0.3, 1.0), seeds=range(8)):
    """Full sweep of EcoFed's shadow price on the evaluation seeds (reported in full, never used for selection)."""
    jobs = []
    for ds in ("uci_har", "pamap2"):
        for phi in (0.25, 1.0):
            best = tuned(ds, phi)["ecofed"]
            for lam in lams:
                kw = dict(best["kw"]); kw["q_floor"] = lam
                for s in seeds:
                    jobs.append(dict(exp="lam_sweep", dataset=ds, fold=s % 4 if ds == "pamap2" else 0, method="ecofed",
                                     kw=kw, cfg=dict(phi=phi, T=T, seed=s, lr=best["lr"])))
    return jobs


def overhead_jobs(T=80):
    """True per-sample overhead (fitted from CPU latency) that the controllers' nominal energy model ignores."""
    jobs = []
    for ds, ov_list in (("pamap2", (0.5e6, 1.0e6)), ("uci_har", (0.5e6, 1.0e6))):
        best = tuned(ds, 0.25)
        for ov in ov_list:
            for m in ("fedavg", "oort", "energy_greedy", "static_depth", "fedavg_paced", "ecofed"):
                for s in SEEDS:
                    jobs.append(dict(exp="sens_ovh", dataset=ds, fold=s % 4 if ds == "pamap2" else 0, method=m,
                                     kw=best[m]["kw"], cfg=dict(phi=0.25, T=T, seed=s, lr=best[m]["lr"], overhead_macs=ov)))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "main":
        for ds in ("uci_har", "pamap2"):
            for phi in (0.1, 0.25, 1.0):
                jobs += main_jobs(ds, phi)
    elif which == "lam":
        jobs = lambda_sweep_jobs()
    elif which == "ovh":
        jobs = overhead_jobs()
    elif which == "extra":
        jobs = extra_seed_jobs()
    elif which == "sens":
        jobs = sens_jobs()
    print(len(jobs), "jobs", flush=True)
    run_all(jobs)
    print("done")
