"""Build all result figures and LaTeX tables from results/raw/*. Usage: python -m analysis.make_results"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import aggregate as AG
from analysis.style import (DATASET_LABEL, GRID, INK, INK2, METHOD_COLOR, METHOD_LABEL, SLOT, save, setup)

plt = setup()
TAB = Path("paper/tables"); TAB.mkdir(parents=True, exist_ok=True)
MAIN = ["ecofed", "oort", "energy_greedy", "static_depth", "fedavg_paced", "fedavg_q8", "fedprox", "fedavg"]
CURVE_METHODS = ["ecofed", "oort", "energy_greedy", "static_depth", "fedavg_paced", "fedavg"]
PHIS = (0.1, 0.25, 1.0)


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def mean_curve(rs, grid):
    ys = []
    for r in rs:
        x = np.array(r["log"]["cum_energy"]) / r["fleet_capacity"]
        va = np.array(r["log"]["val_acc"])
        y = np.array([a[int(np.argmax(v))] for a, v in zip(r["log"]["acc"], va)])
        ys.append(np.interp(grid, x, y, left=y[0]))
    ys = np.array(ys)
    return ys.mean(0), np.array([ci95(ys[:, i]) for i in range(ys.shape[1])])


def fmt(m, c=None, d=3, pct=False):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    s = f"{m*100:.1f}" if pct else f"{m:.{d}f}"
    return s if c is None or np.isnan(c) else f"{s}\\,$\\pm$\\,{(c*100 if pct else c):.{1 if pct else d}f}"


# ------------------------------------------------------------------ figures
def fig_curves(df):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True)
    grid = np.linspace(0.0, 1.0, 201)
    for i, ds in enumerate(("uci_har", "pamap2")):
        for j, phi in enumerate(PHIS):
            ax = axes[i, j]
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            for m in CURVE_METHODS:
                rs = list(sub[sub.method == m]["_res"])
                if not rs:
                    continue
                mu, ci = mean_curve(rs, grid)
                lw = 2.2 if m == "ecofed" else 1.3
                ax.plot(grid, mu, color=METHOD_COLOR[m], lw=lw, label=METHOD_LABEL[m], zorder=5 if m == "ecofed" else 3)
                if m == "ecofed":
                    ax.fill_between(grid, mu - ci, mu + ci, color=METHOD_COLOR[m], alpha=0.15, lw=0)
            ax.set_title(f"{DATASET_LABEL[ds]}, $\\phi={phi}$", fontsize=8, loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel("test accuracy (best exit)")
            if i == 1:
                ax.set_xlabel("fraction of fleet capacity spent")
    axes[0, 0].legend(loc="lower right", ncol=1, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_accuracy_vs_energy")


def fig_eta(df):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    k = 0
    for ds in ("uci_har", "pamap2"):
        for phi in (0.25, 1.0):
            ax = axes[k]; k += 1
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            ys, labs = [], []
            for yi, m in enumerate(MAIN[::-1]):
                v = sub[sub.method == m]["eta90"].values
                ok = v[~np.isnan(v)]
                if len(ok):
                    ax.errorbar(ok.mean(), yi, xerr=ci95(ok) if len(ok) > 1 else 0, fmt="o", color=METHOD_COLOR[m],
                                ms=4.5 if m != "ecofed" else 6, capsize=2, lw=1)
                ax.text(1.02, yi, f"{len(ok)}/{len(v)}", transform=ax.get_yaxis_transform(), fontsize=6, color=INK2, va="center")
                labs.append(METHOD_LABEL[m])
            ax.set_yticks(range(len(MAIN))); ax.set_yticklabels(labs if k == 1 else [""] * len(MAIN), fontsize=7)
            ax.set_title(f"{DATASET_LABEL[ds]}, $\\phi={phi}$", fontsize=8, loc="left", fontweight="bold")
            ax.set_xlabel("energy to 90% of plateau (J)", fontsize=7.5)
            ax.set_xscale("log")
            from matplotlib.ticker import NullFormatter, FuncFormatter
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            lo, hi = ax.get_xlim(); ax.set_xticks([t for t in (5, 10, 20, 30, 50, 100, 200, 300, 500) if lo <= t <= hi])
    fig.tight_layout()
    save(fig, "fig5_eta")


def fig_dots(df, metric, name, xlabel, phis=(0.1, 0.25), pct=False):
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    k = 0
    for ds in ("uci_har", "pamap2"):
        for phi in phis:
            ax = axes[k]; k += 1
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            for yi, m in enumerate(MAIN[::-1]):
                v = sub[sub.method == m][metric].values.astype(float)
                if not len(v):
                    continue
                s = 100 if pct else 1
                ax.errorbar(np.nanmean(v) * s, yi, xerr=ci95(v) * s, fmt="o", color=METHOD_COLOR[m], ms=4.5 if m != "ecofed" else 6, capsize=2, lw=1)
            ax.set_yticks(range(len(MAIN))); ax.set_yticklabels([METHOD_LABEL[m] for m in MAIN[::-1]] if k == 1 else [""] * len(MAIN), fontsize=7)
            ax.set_title(f"{DATASET_LABEL[ds]}, $\\phi={phi}$", fontsize=8, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------ tables
def table_main(df):
    rows = []
    for ds in ("uci_har", "pamap2"):
        for phi in PHIS:
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            best25 = sub.groupby("method").acc25.mean().max()
            best100 = sub.groupby("method").acc100.mean().max()
            bestck = sub.groupby("method").ckpt_acc.mean().max()
            for m in MAIN:
                s = sub[sub.method == m]
                if s.empty:
                    continue
                eta = s["eta90"].values
                ok = eta[~np.isnan(eta)]
                a25, a100 = s.acc25.mean(), s.acc100.mean()
                ck = s.ckpt_acc.mean()
                rows.append(dict(ds=DATASET_LABEL[ds], phi=phi, m=METHOD_LABEL[m], a25=fmt(a25, ci95(s.acc25), pct=True),
                                 a25b=a25 >= best25 - 1e-9, a100=fmt(a100, ci95(s.acc100), pct=True), a100b=a100 >= best100 - 1e-9,
                                 auc=fmt(ck, ci95(s.ckpt_acc), pct=True) + "", ckb=ck >= bestck - 1e-9,
                                 eta=fmt(s.ckpt_E.mean(), None, d=0),
                                 jain=fmt(s.total_E.mean(), None, d=0), dep=fmt(s.depleted.mean(), None, pct=True),
                                 ex4=fmt(s.final_exit4.mean(), None, pct=True)))
    out = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
           "Data / $\\phi$ & Method & Acc@25\\% & Final acc. & Best-ckpt acc. & Ckpt energy (J) & Energy used (J) & Depl.\\ (\\%)\\\\", "\\midrule"]
    last = None
    for r in rows:
        key = (r["ds"], r["phi"])
        lead = f"{r['ds']}, {r['phi']}" if key != last else ""
        if key != last and last is not None:
            out.append("\\midrule")
        last = key
        b = lambda t, flag: f"\\textbf{{{t}}}" if flag else t
        name = f"\\textbf{{{r['m']}}}" if r["m"] == "EcoFed" else r["m"]
        out.append(f"{lead} & {name} & {b(r['a25'], r['a25b'])} & {b(r['a100'], r['a100b'])} & {b(r['auc'], r['ckb'])} & {r['eta']} & {r['jain']} & {r['dep']}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "main.tex").write_text("\n".join(out))


def paired_table(df, metric, fname, baselines=None, scale=100.0, ratio=False):
    """EcoFed vs each baseline, paired by seed. Two-sided paired t-test (on log ratios when ratio=True),
    Holm-corrected within each (dataset, phi) setting across the baselines; bootstrap 95% CI of the mean."""
    baselines = baselines or [m for m in MAIN if m != "ecofed"]
    blocks = []
    for ds in ("uci_har", "pamap2"):
        for phi in PHIS:
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            rows = []
            for b in baselines:
                A = sub[sub.method == "ecofed"].set_index("seed")[metric]
                B = sub[sub.method == b].set_index("seed")[metric]
                idx = A.index.intersection(B.index)
                if ratio:
                    v = (B.loc[idx] / A.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna().values
                    d = np.log(v) if len(v) else v
                else:
                    v = ((A.loc[idx] - B.loc[idx]) * scale).dropna().values
                    d = v
                res = dict(n=len(v), diff=float(np.mean(v)) if len(v) else np.nan, lo=np.nan, hi=np.nan, p=1.0)
                if len(v) >= 3:
                    rng = np.random.default_rng(0)
                    bt = [rng.choice(v, len(v)).mean() for _ in range(4000)]
                    res.update(lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)))
                    res["p"] = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else 1.0
                rows.append((METHOD_LABEL[b], res))
            adj = AG.holm([r["p"] for _, r in rows])
            blocks.append((DATASET_LABEL[ds], phi, rows, adj))
    head = "mean ratio (baseline/EcoFed)" if ratio else "mean $\\Delta$ (pp)"
    out = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
           "Data / $\\phi$ & vs. & " + head + " & 95\\% CI & $n$ & Holm $p$\\\\", "\\midrule"]
    for bi, (ds, phi, rows, adj) in enumerate(blocks):
        if bi:
            out.append("\\midrule")
        for ri, ((b, res), pa) in enumerate(zip(rows, adj)):
            lead = f"{ds}, {phi}" if ri == 0 else ""
            star = "$^{*}$" if pa < 0.05 else ""
            ci = "--" if np.isnan(res["lo"]) else f"[{res['lo']:.2f}, {res['hi']:.2f}]"
            out.append(f"{lead} & {b} & {res['diff']:.2f}{star} & {ci} & {res['n']} & {pa:.3f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / fname).write_text("\n".join(out))


def table_ablation(df):
    abl = ["ecofed", "ecofed_noQ", "ecofed_noZ", "ecofed_fixdepth", "ecofed_fixprec", "ecofed_fixtau"]
    out = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
           "Data / $\\phi$ & Variant & Acc@25\\% & Acc@100\\% & AUC & Exit-4 acc. & mean depth\\\\", "\\midrule"]
    last = None
    for ds in ("uci_har", "pamap2"):
        for phi in (0.1, 0.25):
            sub = df[(df.dataset == ds) & (df.phi == phi)]
            for m in abl:
                s = sub[sub.method == m]
                if s.empty:
                    continue
                key = (ds, phi)
                if key != last and last is not None:
                    out.append("\\midrule")
                lead = f"{DATASET_LABEL[ds]}, {phi}" if key != last else ""
                last = key
                out.append(f"{lead} & {METHOD_LABEL[m]} & {fmt(s.acc25.mean(), ci95(s.acc25), pct=True)} & {fmt(s.acc100.mean(), ci95(s.acc100), pct=True)} & "
                           f"{fmt(s.auc.mean(), ci95(s.auc), pct=True)} & {fmt(s.final_exit4.mean(), ci95(s.final_exit4), pct=True)} & {s.depth.mean():.2f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "ablation.tex").write_text("\n".join(out))


if __name__ == "__main__":
    df = AG.per_run_table("m2")
    df = AG.add_eta(df)
    df.drop(columns=["_res"]).to_csv("results/main_runs.csv", index=False)
    json.dump(df.attrs["ref_plateau"], open("results/ref_plateau.json", "w"))
    fig_curves(df); fig_eta(df)
    fig_dots(df, "depleted", "fig8_depleted", "clients with depleted battery (%)", pct=True)
    table_main(df)
    paired_table(df, "acc25", "paired_acc25.tex")
    paired_table(df, "acc100", "paired_acc100.tex")
    paired_table(df, "ckpt_acc", "paired_ckpt.tex")
    paired_table(df, "total_E", "paired_energy.tex", ratio=True)
    paired_table(df, "eta90", "paired_eta90.tex", ratio=True)
    table_ablation(df)
    print("results built:", len(df), "runs")
