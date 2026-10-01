from pathlib import Path

import numpy as np
import pytest

from data_loaders import uci_har as H

REAL = Path("data/UCI HAR Dataset")


def _write_split(base: Path, split: str, subjects: list[int], per: int, rng):
    sdir = base / split
    (sdir / "Inertial Signals").mkdir(parents=True)
    n = len(subjects) * per
    for sig in ("body_acc", "body_gyro", "total_acc"):
        for ax in "xyz":
            np.savetxt(sdir / "Inertial Signals" / f"{sig}_{ax}_{split}.txt",
                       rng.normal(size=(n, 128)))
    np.savetxt(sdir / f"y_{split}.txt", rng.integers(1, 7, n), fmt="%d")
    np.savetxt(sdir / f"subject_{split}.txt", np.repeat(subjects, per), fmt="%d")


@pytest.fixture()
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "UCI HAR Dataset"
    _write_split(base, "train", [1, 3, 5, 6, 7], 8, rng)
    _write_split(base, "test", [2, 4], 8, rng)
    return tmp_path


def test_shapes_and_labels(fake_root):
    tr, te = H.load_uci_har(fake_root, fetch=False)
    assert tr.X.shape == (40, 9, 128) and te.X.shape == (16, 9, 128)
    assert tr.X.dtype == np.float32
    assert tr.y.min() >= 0 and tr.y.max() <= 5
    assert not set(tr.subjects) & set(te.subjects)


def test_signal_sets(fake_root):
    tr, _ = H.load_uci_har(fake_root, signals="body", fetch=False)
    assert tr.X.shape[1] == 6


def test_missing_data_without_fetch(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.load_uci_har(tmp_path, fetch=False)


def test_standardize_uses_train_stats_only(fake_root):
    tr, te = H.load_uci_har(fake_root, fetch=False)
    tr_s, te_s = H.standardize(tr, te)
    assert np.allclose(tr_s.X.mean(axis=(0, 2)), 0, atol=1e-4)
    assert np.allclose(tr_s.X.std(axis=(0, 2)), 1, atol=1e-3)
    assert not np.allclose(te_s.X.mean(axis=(0, 2)), 0, atol=1e-6)


@pytest.mark.parametrize("hz,length", [(25, 64), (10, 25)])
def test_downsample(fake_root, hz, length):
    tr, _ = H.load_uci_har(fake_root, fetch=False)
    d = H.downsample(tr, hz)
    assert d.X.shape[2] == length and d.hz == (50 // round(50 / hz))
    assert H.downsample(tr, 50) is tr


def test_downsample_preserves_mean_of_constant(fake_root):
    tr, _ = H.load_uci_har(fake_root, fetch=False)
    c = H.HARData(np.full_like(tr.X, 3.0), tr.y, tr.subjects)
    assert np.allclose(H.downsample(c, 10).X, 3.0)


def test_split_by_subject_disjoint(fake_root):
    tr, _ = H.load_uci_har(fake_root, fetch=False)
    a, b = H.split_by_subject(tr, 0.4, seed=1)
    assert len(a) + len(b) == len(tr)
    assert not set(a.subjects) & set(b.subjects)
    c, d = H.split_by_subject(tr, [1, 3])
    assert set(d.subjects) == {1, 3}


def test_federated_partition_one_client_per_subject(fake_root):
    tr, _ = H.load_uci_har(fake_root, fetch=False)
    parts = H.federated_partition(tr)
    assert set(parts) == {1, 3, 5, 6, 7}
    assert all(set(p.subjects) == {s} for s, p in parts.items())


def test_leave_subject_out_folds(fake_root):
    tr, _ = H.load_uci_har(fake_root, fetch=False)
    seen = set()
    for a, b in H.leave_subject_out_folds(tr, n_folds=5):
        assert not set(a.subjects) & set(b.subjects)
        seen |= set(b.subjects)
    assert seen == set(tr.subjects)


@pytest.mark.skipif(not REAL.is_dir(), reason="real dataset not downloaded")
def test_real_dataset_matches_published_stats():
    tr, te = H.load_uci_har("data", fetch=False)
    assert len(tr) == 7352 and len(te) == 2947
    assert len(np.unique(tr.subjects)) == 21 and len(np.unique(te.subjects)) == 9
    assert tr.X.shape[1:] == (9, 128)
    assert set(np.unique(tr.y)) == set(range(6))
    assert not set(tr.subjects) & set(te.subjects)
