"""Coverage-floor, sensitivity, lifecycle and scaling figures/tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import aggregate as AG
from analysis.lifecycle import calibrate_and_test, exit_energies
from analysis.make_results import MAIN, TAB, ci95, fmt
from analysis.style import (DATASETS, DATASET_LABEL, INK, INK2, METHOD_COLOR, METHOD_LABEL, SLOT, save, setup)
from experiments.common import load_results
from fl.tasks import build_task

plt = setup()


# ------------------------------------------------------------------ coverage floor
def coverage_figure():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    summary = {}
    for ax, ds in zip(axes[:2], ("uci_har", "pamap2")):
        R = load_results(f"cov_{ds}")
        if not R:
            continue
        df = pd.DataFrame([dict(rho=j["kw"]["rho"], seed=j["cfg"]["seed"], g=r["gradnorm_sq"],
                                a4=r["log"]["acc"][-1][-1], a1=r["log"]["acc"][-1][0]) for j, r in R])
        g = df.groupby("rho").agg(g=("g", "mean"), gc=("g", ci95), a4=("a4", "mean"), a4c=("a4", ci95)).reset_index()
        x = (1 - g.rho.values) ** 2
        ax.errorbar(x, g.g.values, yerr=g.gc.values, fmt="o", color=SLOT["blue"], capsize=2, lw=1)
        sl, ic, r, p, _ = stats.linregress((1 - df.rho.values) ** 2, df.g.values)
        xs = np.linspace(0, x.max(), 20)
        ax.plot(xs, ic + sl * xs, color=SLOT["blue"], lw=1, alpha=0.6)
        rho_s, p_s = stats.spearmanr((1 - df.rho.values) ** 2, df.g.values)
        ax.set_title(f"{DATASET_LABEL[ds]}", fontsize=8, loc="left", fontweight="bold")
        ax.set_xlabel("$(1-\\sigma)^2$"); ax.set_ylabel("$\\|\\nabla F(\\mathbf{w}^T)\\|^2$")
        ax.text(0.04, 0.93, f"$R^2$={r**2:.2f}, Spearman $\\rho$={rho_s:.2f}", transform=ax.transAxes, fontsize=7, va="top")
        summary[ds] = dict(r2=r ** 2, spearman=float(rho_s), p=float(p_s), slope=sl, intercept=ic,
                           n=len(df), by_rho=g.to_dict("list"))
    ax = axes[2]
    for ds, col in (("uci_har", SLOT["blue"]), ("pamap2", SLOT["orange"])):
        R = load_results(f"cov_{ds}")
        if not R:
            continue
        df = pd.DataFrame([dict(rho=j["kw"]["rho"], a4=r["log"]["acc"][-1][-1]) for j, r in R])
        g = df.groupby("rho").a4.agg(["mean", ci95]).reset_index()
        ax.errorbar(g.rho, g["mean"], yerr=g["ci95"], color=col, marker="o", capsize=2, lw=1.3, label=DATASET_LABEL[ds])
    ax.set_xlabel("coverage $\\sigma$ of blocks $2..L$"); ax.set_ylabel("exit-4 test accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout(); save(fig, "fig6_coverage_floor")
    json.dump(summary, open("results/coverage_floor_summary.json", "w"), indent=1, default=float)
    return summary


# ------------------------------------------------------------------ sensitivity
def sensitivity(main_df):
    out = {}
    Rr = load_results("s3_radio")
    if Rr:
        df = pd.DataFrame([dict(method=j["method"], radio=j["cfg"]["radio_scale"], seed=j["cfg"]["seed"],
                                auc=AG_auc(r), acc25=AG_acc(r, 0.25), total=r["total_energy"] / r["fleet_capacity"]) for j, r in Rr])
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
        for m in ("ecofed", "oort", "energy_greedy", "fedavg_q8", "fedavg"):
            g = df[df.method == m].groupby("radio").auc.agg(["mean", ci95]).reset_index()
            axes[0].errorbar(g.radio, g["mean"] * 100, yerr=g["ci95"] * 100, color=METHOD_COLOR[m], marker="o", capsize=2,
                             lw=2.0 if m == "ecofed" else 1.2, label=METHOD_LABEL[m])
        axes[0].set_xscale("log"); axes[0].set_xlabel("radio energy scale"); axes[0].set_ylabel("AUC of accuracy vs energy (%)")
        axes[0].set_title("(a) Radio-to-compute cost ratio", fontsize=8, fontweight="bold", loc="left")
        out["radio"] = df.groupby(["method", "radio"]).auc.mean().unstack().to_dict()
    Re = load_results("s3_err")
    if Re:
        df = pd.DataFrame([dict(method=j["method"], err=j["cfg"]["energy_err"], seed=j["cfg"]["seed"], auc=AG_auc(r)) for j, r in Re])
        base = main_df[(main_df.dataset == "pamap2") & (main_df.phi == 0.25) & (main_df.seed < 8)][["method", "seed", "auc"]].assign(err=0.0)
        df = pd.concat([df, base[base.method.isin(df.method.unique())]])
        ax = axes[1]
        for m in ("ecofed", "static_depth", "oort", "energy_greedy", "fedavg_paced"):
            g = df[df.method == m].groupby("err").auc.agg(["mean", ci95]).reset_index()
            ax.errorbar(g.err, g["mean"] * 100, yerr=g["ci95"] * 100, color=METHOD_COLOR[m], marker="o", capsize=2,
                        lw=2.0 if m == "ecofed" else 1.2, label=METHOD_LABEL[m])
        ax.set_xlabel("std of log-error"); ax.set_ylabel("AUC of accuracy vs energy (%)")
        ax.set_title("(b) Energy-model misspecification", fontsize=8, fontweight="bold", loc="left")
        out["err"] = df.groupby(["method", "err"]).auc.mean().unstack().to_dict()
    Ro = load_results("s3_ovh")
    if Ro:
        ax = axes[2]
        do = pd.DataFrame([dict(method=j["method"], ds=j["dataset"], ov=j["cfg"]["overhead_macs"], seed=j["cfg"]["seed"], auc=AG_auc(r)) for j, r in Ro])
        base = main_df[(main_df.dataset == "pamap2") & (main_df.phi == 0.25) & (main_df.seed < 8)][["method", "seed", "auc"]].assign(ov=0.0, ds="pamap2")
        dd = pd.concat([do[do.ds == "pamap2"], base[base.method.isin(do.method.unique())]])
        for m in ("ecofed", "static_depth", "oort", "energy_greedy", "fedavg_paced", "fedavg"):
            g = dd[dd.method == m].groupby("ov").auc.agg(["mean", ci95]).reset_index()
            ax.errorbar(g.ov / 1e6, g["mean"] * 100, yerr=g["ci95"] * 100, color=METHOD_COLOR[m], marker="o", capsize=2,
                        lw=2.0 if m == "ecofed" else 1.2, label=METHOD_LABEL[m])
        ax.set_xlabel("overhead ($10^6$ MAC-eq./sample)"); ax.set_ylabel("AUC (%)")
        ax.set_title("(c) Per-sample overhead", fontsize=8, fontweight="bold", loc="left")
        out["ovh"] = {ds: do[do.ds == ds].groupby(["method", "ov"]).auc.mean().unstack().to_dict() for ds in ("pamap2", "uci_har")}
    if Rr or Re:
        hs = [plt.Line2D([], [], color=METHOD_COLOR[m], marker="o", lw=2.0 if m == "ecofed" else 1.2, label=METHOD_LABEL[m])
              for m in ("ecofed", "static_depth", "oort", "energy_greedy", "fedavg_paced", "fedavg_q8", "fedavg")]
        fig.legend(handles=hs, loc="lower center", ncol=7, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "fig7_sensitivity")
    json.dump(out, open("results/sensitivity_summary.json", "w"), indent=1, default=float)
    return out


def AG_auc(r):
    from analysis.metrics import auc_energy
    return auc_energy(r, "acc", "best")


def AG_acc(r, f):
    from analysis.metrics import acc_at_energy
    return acc_at_energy(r, f, "acc", "best")


# ------------------------------------------------------------------ lifecycle
def lifecycle(phi_list=(0.25, 1.0), delta=0.01):
    rows = []
    tasks = {}
    for job, r in load_results("m3"):
        phi = job["cfg"]["phi"]
        if phi not in phi_list or job["method"] not in ("ecofed", "fedavg", "fedavg_single", "oort", "energy_greedy", "static_depth"):
            continue
        ds, fold = job["dataset"], job.get("fold", 0)
        if (ds, fold) not in tasks:
            tasks[(ds, fold)] = build_task(ds, fold)
        t = tasks[(ds, fold)]
        e_exit = exit_energies(t.in_ch, t.n_classes, t.in_len)
        allowed = [3] if job["method"] == "fedavg_single" else None
        res = calibrate_and_test(r["val_probs"], t.val[1].numpy(), r["test_probs"], t.test[1].numpy(), e_exit, delta, allowed)
        rows.append(dict(dataset=ds, phi=phi, method=job["method"], seed=job["cfg"]["seed"], e_train=r["total_energy"],
                         e_exit4=float(e_exit[3]), **{k: v for k, v in res.items() if k != "exit_frac"},
                         exit1=float(res["exit_frac"][0]), exit4=float(res["exit_frac"][3])))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["crossover"] = df.e_train / df.cascade_energy
    df.to_csv("results/lifecycle_runs.csv", index=False)
    return df


def lifecycle_outputs(df):
    TAB.mkdir(exist_ok=True)
    n_dev_of = {"uci_har": 16, "pamap2": 20, "speech": 24}
    rate_of = {"uci_har": 1 / 1.28, "pamap2": 1 / 1.28, "speech": 1 / 0.5}      # windows per second per device
    out = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
           "Data / $\\phi$ & Model & Test acc. (\\%) & $E_{train}$ (J) & $\\bar e_{inf}$ ($\\mu$J) & exits at 1 (\\%) & $M_\\times$ ($10^5$) & days to $M_\\times$\\\\", "\\midrule"]
    last = None
    for ds in DATASETS:
        for phi in (0.25, 1.0):
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            if sub.empty:
                continue
            n_dev = n_dev_of[ds]
            for m, lab in (("fedavg_single", "FedAvg, single exit"), ("fedavg", "FedAvg, cascade"), ("ecofed", "EcoFed, cascade")):
                s = sub[sub.method == m]
                if s.empty:
                    continue
                key = (ds, phi)
                if key != last and last is not None:
                    out.append("\\midrule")
                lead = f"{DATASET_LABEL[ds]}, {phi}" if key != last else ""
                last = key
                acc = s.cascade_acc.mean() * 100
                e_inf = s.cascade_energy.mean()
                xo = s.e_train.mean() / e_inf
                days = xo / (n_dev * 24 * 3600 * rate_of[ds])      # always-on, one window per 1/rate seconds per device
                out.append(f"{lead} & {lab} & {acc:.1f} & {s.e_train.mean():.1f} & {e_inf*1e6:.0f} & {s.exit1.mean()*100:.0f} & {xo/1e5:.2f} & {days:.2f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "lifecycle.tex").write_text("\n".join(out))
    # figure: lifecycle energy vs number of inferences
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
    M = np.logspace(3, 8, 100)
    for ax, ds in zip(axes, DATASETS):
        sub = df[(df.dataset == ds) & (df.phi == 1.0)]
        for m, lab, col, ls in (("fedavg_single", "FedAvg, single exit", METHOD_COLOR["fedavg_single"], "--"),
                                ("fedavg", "FedAvg, cascade", METHOD_COLOR["fedavg"], "-"),
                                ("ecofed", "EcoFed, cascade", METHOD_COLOR["ecofed"], "-")):
            s = sub[sub.method == m]
            if s.empty:
                continue
            ax.plot(M, s.e_train.mean() + M * s.cascade_energy.mean(), color=col, ls=ls, lw=2.0 if m == "ecofed" else 1.3, label=lab)
            ax.plot([s.e_train.mean() / s.cascade_energy.mean()], [2 * s.e_train.mean()], "o", color=col, ms=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{DATASET_LABEL[ds]}, $\\phi=1$", fontsize=8, loc="left", fontweight="bold")
        ax.set_xlabel("inferences served by the fleet $M$")
    axes[0].set_ylabel("lifecycle energy (J)")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig9_lifecycle")


def lambda_figure(main_df):
    R = load_results("s3_lam")
    if not R:
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.4))
    axes = axes.T.reshape(-1)
    k = 0
    summary = {}
    for ds in DATASETS:
        for phi in (0.25, 1.0):
            ax = axes[k]; k += 1
            sub = main_df[(main_df.dataset == ds) & (main_df.phi == phi) & (main_df.seed < 8)]
            for m in MAIN:
                if m == "ecofed":
                    continue
                s_ = sub[sub.method == m]
                ax.errorbar(s_.ckpt_E.mean(), s_.ckpt_acc.mean() * 100, color=METHOD_COLOR[m], marker="o", ms=4, lw=0)
            rows = [dict(lam=j["kw"]["q_floor"], acc=AG_ckpt(r)[0], e=AG_ckpt(r)[1], seed=j["cfg"]["seed"])
                    for j, r in R if j["dataset"] == ds and j["cfg"]["phi"] == phi]
            d = pd.DataFrame(rows)
            g = d.groupby("lam").agg(acc=("acc", "mean"), e=("e", "mean")).reset_index()
            ax.plot(g.e, g.acc * 100, color=METHOD_COLOR["ecofed"], marker="o", lw=1.8, ms=4.5, zorder=5)
            for _, r in g.iterrows():
                ax.annotate(f"{r.lam:g}", (r.e, r.acc * 100), textcoords="offset points", xytext=(3, 3), fontsize=6, color=METHOD_COLOR["ecofed"])
            ax.set_title(f"{DATASET_LABEL[ds]}, $\\phi={phi}$", fontsize=8, loc="left", fontweight="bold")
            ax.set_xlabel("energy to checkpoint (J)")
            if k == 1:
                ax.set_ylabel("checkpoint test accuracy (%)")
            summary[f"{ds}_{phi}"] = g.to_dict("list")
    h = [plt.Line2D([], [], color=METHOD_COLOR["ecofed"], marker="o", label="EcoFed, shadow price $\\lambda$ (labels)")]
    h += [plt.Line2D([], [], color=METHOD_COLOR[m], marker="o", lw=0, label=METHOD_LABEL[m]) for m in MAIN if m != "ecofed"]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "fig10_pareto")
    json.dump(summary, open("results/lambda_sweep_summary.json", "w"), indent=1, default=float)


def exit_table(main_df):
    out = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Data / $\\phi$ & Method & Exit 1 & Exit 2 & Exit 3 & Exit 4\\\\", "\\midrule"]
    last = None
    for ds in DATASETS:
        for phi in (0.25,):
            sub = main_df[(main_df.dataset == ds) & (main_df.phi == phi)]
            for m in ("ecofed", "static_depth", "oort", "fedavg", "fedavg_single"):
                rs = list(sub[sub.method == m]["_res"])
                acc = np.array([np.mean([r["log"]["acc"][-3:][i][j] for i in range(3)]) for r in rs for j in range(4)]).reshape(len(rs), 4) * 100
                key = (ds, phi)
                if key != last and last is not None:
                    out.append("\\midrule")
                lead = f"{DATASET_LABEL[ds]}, {phi}" if key != last else ""
                last = key
                cells = " & ".join(f"{acc[:, j].mean():.1f}" for j in range(4))
                out.append(f"{lead} & {METHOD_LABEL[m]} & {cells}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "exits.tex").write_text("\n".join(out))


def system_figure():
    sc = json.load(open("results/scaling.json"))
    lat = json.load(open("results/latency_validation.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    for m in ("ecofed", "oort"):
        d = [r for r in sc if r["method"] == m]
        ax.plot([r["N"] for r in d], [r["seconds"] for r in d], color=METHOD_COLOR[m], marker="o", lw=2.0 if m == "ecofed" else 1.3, label=METHOD_LABEL[m])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("clients $N$"); ax.set_ylabel("selection time per round (s)")
    ax.set_title("(a) Controller overhead", fontsize=8, fontweight="bold", loc="left"); ax.legend()
    ax = axes[1]
    for ds, col in (("uci_har", SLOT["blue"]), ("pamap2", SLOT["orange"])):
        v = lat[ds]
        ax.plot(v["rel_cost_model"], v["rel_time"], marker="o", color=col, label=f"{DATASET_LABEL[ds]} (depth 1..4)")
    mx = max(max(lat[d]["rel_cost_model"]) for d in lat)
    ax.plot([1, mx], [1, mx], color=INK2, ls="--", lw=1, label="proportional to MACs")
    ax.set_xlabel("modelled cost relative to depth 1"); ax.set_ylabel("measured CPU time relative to depth 1")
    ax.set_title("(b) Cost model vs. measured latency", fontsize=8, fontweight="bold", loc="left"); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "fig11_system")


def AG_ckpt(r):
    c = AG.checkpoint(r)
    return c["ckpt_acc"], c["ckpt_E"]


if __name__ == "__main__":
    main_df = AG.per_run_table("m3")
    coverage_figure()
    sensitivity(main_df)
    lambda_figure(main_df)
    system_figure()
    exit_table(main_df)
    ldf = lifecycle()
    if not ldf.empty:
        lifecycle_outputs(ldf)
    print("extra outputs built")
