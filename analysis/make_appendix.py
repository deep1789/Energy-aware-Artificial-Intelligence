"""Appendix material: tuned hyper-parameters table, class distribution per client."""
import json
import numpy as np
from analysis.make_results import MAIN, TAB
from analysis.style import DATASETS, DATASET_LABEL, METHOD_LABEL, SLOT, save, setup
from fl.tasks import build_task

plt = setup()


def hparam_table():
    out = ["\\begin{tabular}{@{}llllc@{}}", "\\toprule", "Data / $\\phi$ & Method & Knobs & lr & Validation AUC (\\%)\\\\", "\\midrule"]
    last = None
    for ds in DATASETS:
        for phi in (0.1, 0.25):
            best = json.load(open(f"results/best4_{ds}_{phi}.json"))
            for m in MAIN:
                d = best[m]
                kw = ", ".join(f"{k}={v}" for k, v in d["kw"].items()) or "--"
                kw = kw.replace("_", "\\_")
                key = (ds, phi)
                if key != last and last is not None:
                    out.append("\\midrule")
                lead = f"{DATASET_LABEL[ds]}, {phi}" if key != last else ""
                last = key
                out.append(f"{lead} & {METHOD_LABEL[m]} & {kw} & {d['lr']} & {d['val_auc']*100:.1f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "hparams.tex").write_text("\n".join(out))


def noniid_figure():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    for ax, ds in zip(axes, DATASETS):
        t = build_task(ds, 0)
        M = np.array([np.bincount(y.numpy(), minlength=t.n_classes) / len(y) for _, y in t.clients.values()])
        im = ax.imshow(M.T, aspect="auto", cmap="Blues", vmin=0, vmax=0.6)
        ax.set_title(DATASET_LABEL[ds], fontsize=8, fontweight="bold", loc="left")
        ax.set_xlabel("client"); ax.set_ylabel("activity class"); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="share of client windows")
    save(fig, "fig_a1_noniid")


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    noniid_figure()
    hparam_table()
    print("appendix done")
