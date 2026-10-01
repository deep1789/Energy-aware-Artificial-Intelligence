"""UCI HAR loader (raw inertial signals) with subject-wise splits for federated use.

The dataset has 30 subjects, 6 activities, 9 inertial channels sampled at 50 Hz and cut
into 128-sample windows with 50% overlap. Overlap means random window-level splits leak
between train and test, so every split helper here works at subject level.
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")
ACTIVITIES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING")
SIGNAL_SETS = {
    "all": ("body_acc", "body_gyro", "total_acc"),
    "body": ("body_acc", "body_gyro"),
    "total": ("total_acc", "body_gyro"),
}
AXES = ("x", "y", "z")
NATIVE_HZ = 50


@dataclass
class HARData:
    X: np.ndarray          # (N, C, T) float32
    y: np.ndarray          # (N,) int64 in [0, 6)
    subjects: np.ndarray   # (N,) int64 subject ids (1..30)
    hz: int = NATIVE_HZ

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "HARData":
        return HARData(self.X[mask], self.y[mask], self.subjects[mask], self.hz)


def download(root: str | Path) -> Path:
    """Download and extract the dataset into ``root``; returns the dataset directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "UCI HAR Dataset"
    if target.is_dir():
        return target
    outer = root / "har.zip"
    if not outer.exists():
        urllib.request.urlretrieve(URL, outer)
    with zipfile.ZipFile(outer) as z:
        z.extractall(root)
    inner = root / "UCI HAR Dataset.zip"
    if inner.exists():
        with zipfile.ZipFile(inner) as z:
            z.extractall(root)
    if not target.is_dir():
        raise FileNotFoundError(f"Extraction did not produce {target}")
    return target


def _load_split(base: Path, split: str, signals: tuple[str, ...]) -> HARData:
    sdir = base / split
    channels = []
    for sig in signals:
        for ax in AXES:
            channels.append(np.loadtxt(sdir / "Inertial Signals" / f"{sig}_{ax}_{split}.txt",
                                       dtype=np.float32))
    X = np.stack(channels, axis=1)                                   # (N, C, 128)
    y = np.loadtxt(sdir / f"y_{split}.txt", dtype=np.int64) - 1      # labels are 1-based
    subj = np.loadtxt(sdir / f"subject_{split}.txt", dtype=np.int64)
    if not (len(X) == len(y) == len(subj)):
        raise ValueError(f"Inconsistent row counts in {sdir}")
    return HARData(X, y, subj)


def load_uci_har(root: str | Path = "data", signals: str = "all",
                 fetch: bool = True) -> tuple[HARData, HARData]:
    """Return the official (train, test) split. Train and test use disjoint subjects."""
    if signals not in SIGNAL_SETS:
        raise ValueError(f"signals must be one of {sorted(SIGNAL_SETS)}")
    root = Path(root)
    base = root / "UCI HAR Dataset"
    if not base.is_dir():
        if not fetch:
            raise FileNotFoundError(f"{base} not found (fetch=False)")
        base = download(root)
    sig = SIGNAL_SETS[signals]
    return _load_split(base, "train", sig), _load_split(base, "test", sig)


def standardize(train: HARData, *others: HARData) -> tuple[HARData, ...]:
    """Per-channel z-scoring with statistics from ``train`` only (no test leakage)."""
    mean = train.X.mean(axis=(0, 2), keepdims=True)
    std = train.X.std(axis=(0, 2), keepdims=True) + 1e-8
    out = []
    for d in (train, *others):
        out.append(HARData(((d.X - mean) / std).astype(np.float32), d.y, d.subjects, d.hz))
    return tuple(out)


def downsample(data: HARData, hz: int) -> HARData:
    """Reduce sampling rate by block-averaging (a cheap anti-aliasing filter).

    ``hz`` must give an integer factor against the current rate or the nearest
    integer factor is used; trailing samples that do not fill a block are dropped.
    Window duration is preserved up to that truncation.
    """
    if hz > data.hz or hz <= 0:
        raise ValueError("hz must be in (0, current rate]")
    f = max(1, round(data.hz / hz))
    if f == 1:
        return data
    T = (data.X.shape[2] // f) * f
    X = data.X[:, :, :T].reshape(len(data), data.X.shape[1], T // f, f).mean(axis=3)
    return HARData(X.astype(np.float32), data.y, data.subjects, data.hz // f)


def split_by_subject(data: HARData, val_subjects: list[int] | float = 0.2,
                     seed: int = 0) -> tuple[HARData, HARData]:
    """Split into (train, val) with disjoint subjects.

    ``val_subjects`` is either an explicit list of subject ids or a fraction.
    """
    ids = np.unique(data.subjects)
    if isinstance(val_subjects, float):
        rng = np.random.default_rng(seed)
        k = max(1, int(round(len(ids) * float(val_subjects))))
        val_ids = rng.choice(ids, size=k, replace=False)
    else:
        val_ids = np.asarray(val_subjects)
    mask = np.isin(data.subjects, val_ids)
    return data.subset(~mask), data.subset(mask)


def federated_partition(data: HARData) -> dict[int, HARData]:
    """One client per subject, the natural non-IID partition."""
    return {int(s): data.subset(data.subjects == s) for s in np.unique(data.subjects)}


def leave_subject_out_folds(data: HARData, n_folds: int = 5, seed: int = 0):
    """Yield (train, held_out) pairs where held-out subjects never appear in train."""
    ids = np.unique(data.subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    for chunk in np.array_split(ids, n_folds):
        mask = np.isin(data.subjects, chunk)
        yield data.subset(~mask), data.subset(mask)
