"""Figures that do not depend on experiment outcomes: system overview, coverage schematic, energy decomposition."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from analysis.style import setup, save, SLOT, INK, INK2, GRID, DATASET_LABEL
from energy.model import DEVICE_PROFILES, EnergyModel
from models.multiexit import MultiExitCNN1D

plt = setup()


def box(ax, xy, w, h, text, fc="#f4f3ee", ec=INK2, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, a, b, text=None, color=INK2, rad=0.0, fs=6.8, off=(0, 0.025)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", fontsize=fs, color=INK2)


def fig_overview():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    # server
    box(ax, (0.30, 0.60), 0.40, 0.34, "", fc="#eaf1fb", ec=SLOT["blue"])
    ax.text(0.50, 0.915, "Server", ha="center", fontsize=8, fontweight="bold", color=SLOT["blue"])
    box(ax, (0.32, 0.72), 0.17, 0.14, "EcoFed controller\nscore $s_k(c)$, top-$m$", fc="white", ec=SLOT["blue"])
    box(ax, (0.51, 0.72), 0.17, 0.14, "block-wise\naggregation", fc="white", ec=SLOT["blue"])
    box(ax, (0.32, 0.62), 0.17, 0.08, "queues $q_k$, $Z_j$", fc="white", ec=SLOT["blue"], fs=7)
    box(ax, (0.51, 0.62), 0.17, 0.08, "global multi-exit\nmodel", fc="white", ec=SLOT["blue"], fs=6.5)
    # clients
    for i, (nm, x) in enumerate([("weak device\n(Pi-class)", 0.03), ("mid device\n(Nano-class)", 0.37), ("strong device\n(Orin-class)", 0.71)]):
        box(ax, (x, 0.02), 0.26, 0.42, "", fc="#fbf3ec", ec=SLOT["orange"])
        ax.text(x + 0.13, 0.385, nm.replace("\n", " "), ha="center", va="center", fontsize=7.0, fontweight="bold", color=SLOT["orange"])
        box(ax, (x + 0.02, 0.05), 0.22, 0.08, "battery $B_k$, budget $\\bar E_k$", fc="white", ec=SLOT["orange"], fs=6.8)
        box(ax, (x + 0.02, 0.16), 0.22, 0.14, "train blocks $1..d$,\n$\\tau$ epochs, quantise\nto $b$ bits", fc="white", ec=SLOT["orange"], fs=6.5)
        arrow(ax, (x + 0.13, 0.44), (0.40 + 0.10 * i, 0.60), color=SLOT["orange"])
    ax.text(0.50, 0.50, "reports: loss, battery, measured energy  ↑     configuration $(d,b,\\tau)$ + blocks $1..d$  ↓",
            ha="center", fontsize=6.8, color=INK2)
    # lifecycle
    box(ax, (0.03, 0.62), 0.22, 0.30, "", fc="#eef7f2", ec=SLOT["aqua"])
    ax.text(0.14, 0.895, "Deployment", ha="center", fontsize=8, fontweight="bold", color=SLOT["aqua"])
    box(ax, (0.045, 0.70), 0.19, 0.12, "confidence-gated\nearly-exit cascade", fc="white", ec=SLOT["aqua"], fs=7)
    box(ax, (0.045, 0.635), 0.19, 0.045, "$E_{life}=E_{train}+M\\bar e_{inf}$", fc="white", ec=SLOT["aqua"], fs=7)
    arrow(ax, (0.30, 0.76), (0.25, 0.76), color=SLOT["aqua"])
    fig.savefig  # keep linter quiet
    save(fig, "fig1_overview")


def fig_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off"); ax.grid(False)
    ax.text(0.0, 4.35, "(a) Multi-exit network and who trains what", fontsize=8, fontweight="bold")
    xs = [1.0, 3.0, 5.0, 7.0]
    for j, x in enumerate(xs):
        ax.add_patch(FancyBboxPatch((x, 2.6), 1.5, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#eaf1fb", ec=SLOT["blue"], lw=0.9))
        ax.text(x + 0.75, 3.05, f"block {j+1}", ha="center", va="center", fontsize=7.5)
        ax.add_patch(FancyBboxPatch((x + 0.25, 1.75), 1.0, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fbf3ec", ec=SLOT["orange"], lw=0.8))
        ax.text(x + 0.75, 2.0, f"exit {j+1}", ha="center", va="center", fontsize=7)
        ax.annotate("", xy=(x + 0.75, 2.27), xytext=(x + 0.75, 2.58), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2))
        if j < 3:
            ax.annotate("", xy=(xs[j + 1] - 0.02, 3.05), xytext=(x + 1.52, 3.05), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2))
    rows = [("depth 4 (strong)", 4), ("depth 2 (mid)", 2), ("depth 1 (weak)", 1)]
    for r, (lab, d) in enumerate(rows):
        y = 1.1 - 0.42 * r
        ax.text(0.9, y, lab, ha="right", va="center", fontsize=6.8, color=INK2)
        for j, x in enumerate(xs):
            on = j < d
            ax.add_patch(FancyBboxPatch((x, y - 0.14), 1.5, 0.28, boxstyle="round,pad=0.0,rounding_size=0.06",
                                        fc=SLOT["blue"] if on else "#f0efe9", ec="none", alpha=0.85 if on else 1))
    ax.text(5.0, -0.35, "bars: blocks trained by a client of that depth", ha="center", fontsize=6.8, color=INK2)
    ax = axes[1]
    sig = np.linspace(0.1, 1, 50)
    L = 4
    floor = (L - 1) * (2 * L - 1) / L * (1 - sig) ** 2
    ax.plot(1 - sig, floor, color=SLOT["blue"])
    ax.set_xlabel("missing coverage $1-\\sigma$ of blocks $2..L$")
    ax.set_ylabel("error floor $6B^2/G^2$")
    ax.set_title("(b) Floor of Theorem 1 ($L=4$, equal exit weights)", fontsize=8, fontweight="bold", loc="left")
    ax.text(0.04, 3.3, "$\\frac{(L-1)(2L-1)}{L}(1-\\sigma)^2$", fontsize=8, color=SLOT["blue"])
    save(fig, "fig2_coverage")


def fig_energy():
    m = MultiExitCNN1D(9, 6)
    em = EnergyModel(m.block_stats(), 9, 128)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    names = ["weak", "mid", "strong"]
    cols = [SLOT["blue"], SLOT["orange"], SLOT["aqua"]]
    for ax, nm in zip(axes, names):
        dev = DEVICE_PROFILES[nm]
        d = np.arange(1, 5)
        macs = np.array([dev.alpha * 3 * em.macs(k) * 350 * 2 for k in d])
        mem = np.array([dev.beta * em.bytes_per_sample(k) * 350 * 2 for k in d])
        com32 = np.array([em.e_comm(dev, k, 32) for k in d])
        ax.bar(d, macs, color=SLOT["blue"], label="arithmetic", width=0.62, edgecolor="white", linewidth=1)
        ax.bar(d, mem, bottom=macs, color=SLOT["orange"], label="memory traffic", width=0.62, edgecolor="white", linewidth=1)
        ax.bar(d, com32, bottom=macs + mem, color=SLOT["aqua"], label="radio (32-bit)", width=0.62, edgecolor="white", linewidth=1)
        ax.set_title(f"{nm}", fontsize=8.5, fontweight="bold", loc="left")
        ax.set_xlabel("depth $d$"); ax.set_xticks(d)
        if nm == "weak":
            ax.set_ylabel("J per participation\n($n=350$, $\\tau=2$)")
            ax.legend(loc="upper left")
    save(fig, "fig3_energy_decomposition")


if __name__ == "__main__":
    fig_overview(); fig_coverage(); fig_energy()
    print("figures written")
