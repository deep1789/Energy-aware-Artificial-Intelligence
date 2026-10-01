"""Shared matplotlib style: validated categorical palette (fixed slot order), recessive grid, thin marks."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SLOT = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
            green="#008300", violet="#4a3aa7", red="#e34948")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dc", "#ffffff"

# one fixed colour per entity across every figure
METHOD_COLOR = {
    "ecofed": SLOT["blue"], "oort": SLOT["orange"], "energy_greedy": SLOT["aqua"],
    "static_depth": SLOT["violet"], "fedavg_paced": SLOT["yellow"], "fedavg_q8": SLOT["magenta"],
    "fedprox": SLOT["green"], "fedavg": INK2, "fedavg_single": "#8a8980",
}
METHOD_LABEL = {
    "ecofed": "EcoFed", "oort": "Oort-style", "energy_greedy": "Energy-greedy", "static_depth": "Static-Depth",
    "fedavg_paced": "FedAvg-Paced", "fedavg_q8": "FedAvg-Q8", "fedprox": "FedProx", "fedavg": "FedAvg",
    "fedavg_single": "FedAvg (single exit)",
    "ecofed_noQ": "EcoFed w/o energy queue", "ecofed_noZ": "EcoFed w/o coverage queue",
    "ecofed_fixdepth": "EcoFed, fixed depth", "ecofed_fixprec": "EcoFed, fixed precision",
    "ecofed_fixtau": "EcoFed, fixed epochs",
}
DATASET_LABEL = {"uci_har": "UCI HAR", "pamap2": "PAMAP2"}


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False, "legend.fontsize": 7.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/{name}.{ext}")
    plt.close(fig)
