"""PAMAP2 loader: 3 IMUs (hand, chest, ankle), accelerometer(+-16g) + gyroscope = 18 channels.

Raw 100 Hz is block-averaged to 50 Hz so windows (128 samples = 2.56 s, 50% overlap) match UCI HAR.
Windows are labelled by the majority activity and dropped if the label is transient (id 0) or if
more than 20% of the window is NaN-interpolated. Each subject's recording is split into contiguous
time chunks so that a subject can be turned into several federated clients without sharing windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data_loaders.uci_har import HARData

ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
ID2IDX = {a: i for i, a in enumerate(ACTIVITY_IDS)}
# column indices in the .dat files: acc16 + gyro of hand(3..), chest(20..), ankle(37..)
_BASES = (3, 20, 37)
CHANNELS = [b + off + ax for b in _BASES for off in (1, 7) for ax in range(3)]
SUBJECTS = tuple(range(101, 110))
WIN, STEP = 128, 64


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    idx = np.arange(len(a))
    for c in range(a.shape[1]):
        bad = np.isnan(a[:, c])
        if bad.all():
            a[:, c] = 0.0
        elif bad.any():
            a[bad, c] = np.interp(idx[bad], idx[~bad], a[~bad, c])
    return a


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return windows (n, 18, 128), labels (n,), window start index (n,) for one subject."""
    raw = np.loadtxt(path)
    lab = raw[:, 1].astype(int)
    sig = _interp_nan(raw[:, CHANNELS])
    T = (len(sig) // 2) * 2                       # 100 Hz -> 50 Hz by block mean
    sig = sig[:T].reshape(T // 2, 2, -1).mean(axis=1)
    lab = lab[:T:2]
    X, y, st = [], [], []
    for s in range(0, len(sig) - WIN + 1, STEP):
        l = lab[s:s + WIN]
        vals, cnt = np.unique(l, return_counts=True)
        top = vals[cnt.argmax()]
        if top not in ID2IDX or cnt.max() < 0.9 * WIN:
            continue
        X.append(sig[s:s + WIN].T)
        y.append(ID2IDX[top])
        st.append(s)
    if not X:
        return np.zeros((0, len(CHANNELS), WIN), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.stack(X).astype(np.float32), np.array(y, np.int64), np.array(st, np.int64)


def load_pamap2(root: str | Path = "data", n_chunks: int = 4,
                cache: bool = True) -> tuple[HARData, np.ndarray]:
    """Return (data, chunk_id). ``data.subjects`` holds subject ids (101..109);
    ``chunk_id`` in [0, n_chunks) is the contiguous time chunk of the window within its subject."""
    root = Path(root)
    cpath = root / f"pamap2_cache_{n_chunks}.npz"
    if cache and cpath.exists():
        z = np.load(cpath)
        return HARData(z["X"], z["y"], z["s"]), z["c"]
    base = root / "PAMAP2_Dataset" / "Protocol"
    Xs, ys, ss, cs = [], [], [], []
    for sid in SUBJECTS:
        X, y, st = load_subject(base / f"subject{sid}.dat")
        if len(y) == 0:
            continue
        chunk = np.minimum((np.argsort(np.argsort(st)) * n_chunks) // len(st), n_chunks - 1)
        Xs.append(X); ys.append(y); ss.append(np.full(len(y), sid)); cs.append(chunk)
    X, y, s, c = (np.concatenate(v) for v in (Xs, ys, ss, cs))
    if cache:
        np.savez_compressed(cpath, X=X, y=y, s=s, c=c)
    return HARData(X, y, s.astype(np.int64)), c.astype(np.int64)
