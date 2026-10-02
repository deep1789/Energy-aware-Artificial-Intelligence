# EcoFed: energy-priced, coverage-constrained federated multi-exit learning

Code, experiments and manuscript for a paper prepared for the SUSCOM special issue *Energy-aware Artificial Intelligence*
(`paper/main.tex`, compiled copy `paper/main.pdf`; earlier blueprint in `PAPER_PLAN.md`).

**Important:** all energy figures are *model-based* (assumed device constants, see `energy/model.py`); no power meter was
available. Every claim in the paper is conditional on that, and the paper includes sensitivity studies for it.

## What is here
| Path | Content |
|---|---|
| `data_loaders/` | UCI HAR, PAMAP2 and Speech Commands loaders, person-wise splits (downloaded to `data/`, git-ignored) |
| `models/multiexit.py` | 4-block multi-exit 1D-CNN (27.8k-31.2k parameters depending on the input) |
| `energy/` | analytic energy model; **metered-hardware harness** (`measure_hw.py`, `fit_profile.py`, not run: no meter was available); CPU latency check |
| `fl/` | federated simulator (`sim.py`), baselines and EcoFed (`strategies.py`), tasks (`tasks.py`) |
| `experiments/` | equal-budget tuning (`tune4.py`), main grid and sensitivity jobs (`main.py`), coverage probe, controller scaling |
| `analysis/` | metrics, paired statistics, figures and LaTeX tables |
| `paper/` | LaTeX source, figures, tables, bibliography (references checked against primary sources) |
| `results/` | tuned hyper-parameters (`best4_*.json`), summaries, per-run CSVs (raw logs in `results/raw`, git-ignored) |

## Reproduce
```bash
pip install -r requirements.txt            # plus torch (CPU), scipy, pandas, matplotlib
# datasets: UCI HAR and PAMAP2 (UCI repository) and Speech Commands v0.02 are downloaded/extracted into data/
./run_tuning4.sh                           # equal-budget tuning: 14 random-search trials per method, validation only
cp results/best4_<ds>_0.25.json results/best4_<ds>_1.0.json   # phi = 1 reuses the phi = 0.25 configuration
python -m experiments.main main            # 1,536 jobs (16 seeds; ablations 8 seeds)
python -m experiments.main sens; python -m experiments.main ovh; python -m experiments.main lam
python -m experiments.coverage_floor uci_har 100 6; python -m experiments.coverage_floor pamap2 100 6
python -m energy.validate_latency; python -m experiments.scaling      # run on an idle machine
python -m analysis.make_results; python -m analysis.make_extra; python -m analysis.make_appendix
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Runs are deterministic given the seed and cached by job description, so interrupted sweeps resume (`run_main3.sh` chains the steps above).

## Main findings (see the paper for statistics and caveats; energy is *modelled*, not measured)
* Label-skewed PAMAP2, binding budget (phi = 0.25): final accuracy 57.5 % vs 29-43 % for seven baselines (+15 to +28 pp, Holm-significant); early-stopped accuracy +9 to +17 pp over the energy-blind baselines, tied with a static capability-based depth rule.
* Near-IID UCI HAR: early-stopped accuracy within 1.1 pp of all baselines, up to 2.0x less fleet energy, 1.7-4.2x less energy to a target accuracy than FedAvg-type methods.
* **Speech Commands: EcoFed is worse than several baselines** (2.5-7.8 pp early-stopped accuracy at phi >= 0.25). At the tightest PAMAP2 budget the static depth rule is better when training can be stopped at the best checkpoint.
* Ablations: energy queue and depth control are essential; the coverage queue protects the deep exits; precision control matters only when radio energy is expensive.
* Lifecycle: confidence-gated exits cut inference energy 1.3-7.2x; serving overtakes training energy within 12 minutes to 1.7 days of fleet operation (modelled, always-on).
