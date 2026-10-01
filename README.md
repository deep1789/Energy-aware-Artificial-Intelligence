# Energy-aware Artificial Intelligence

Code and paper plan for an energy-aware federated learning study targeting the SUSCOM special issue
"Energy-aware Artificial Intelligence". See [`PAPER_PLAN.md`](PAPER_PLAN.md) for the full blueprint.

## Layout
| Dir | Purpose | Status |
|---|---|---|
| `data_loaders/` | Dataset loaders and subject-wise partitioners | UCI HAR done |
| `models/` | Multi-exit 1D-CNN, 2D-CNN, DS-CNN, ResNet-8 | todo |
| `energy/` | Power-measurement scripts, energy-model fitting | todo |
| `fl/` | FL baselines and the EcoFed controller | todo |
| `theory/` | Synthetic coverage-floor experiment | todo |
| `experiments/` | YAML configs per experiment | todo |
| `analysis/` | Statistics, figures, tables | todo |
| `paper/` | LaTeX sources | todo |
| `tests/` | pytest suite | UCI HAR tests |

## Quick start
```bash
pip install -r requirements.txt
python -c "from data_loaders.uci_har import load_uci_har; tr, te = load_uci_har('data'); print(len(tr), len(te))"
pytest
```
The first call downloads the dataset (about 60 MB) into `data/`, which is git-ignored.

## UCI HAR loader notes
- Uses the raw inertial signals: 9 channels, 128 samples per window at 50 Hz.
- The official split is by subject (21 train, 9 test). Windows overlap by 50%, so never split at window level.
- `federated_partition` gives one client per subject. `leave_subject_out_folds` and `split_by_subject` keep subjects disjoint.
- `standardize` fits statistics on training data only. `downsample` supports the 25 and 10 Hz sensing-rate experiments.
