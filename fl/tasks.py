"""Federated tasks: client partitions, validation and test sets for UCI HAR and PAMAP2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders import pamap2, uci_har


@dataclass
class Task:
    name: str
    clients: dict[int, tuple[torch.Tensor, torch.Tensor]]
    val: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]
    n_classes: int
    in_ch: int
    in_len: int


def _tensors(d: uci_har.HARData):
    return torch.from_numpy(d.X), torch.from_numpy(d.y)


def build_task(name: str, fold: int = 0, root: str = "data") -> Task:
    if name == "uci_har":
        train, test = uci_har.load_uci_har(root)
        subj = np.unique(train.subjects)
        val_ids = [int(subj[i]) for i in (2, 9, 16)]
        train, test = uci_har.standardize(train, test)
        val = train.subset(np.isin(train.subjects, val_ids))
        tr = train.subset(~np.isin(train.subjects, val_ids))
        parts = uci_har.federated_partition(tr)
        clients = {i: _tensors(p) for i, p in enumerate(parts.values())}
        return Task(name, clients, _tensors(val), _tensors(test), 6, 9, 128)
    if name == "pamap2":
        data, chunk = pamap2.load_pamap2(root, n_chunks=4)
        keep = data.subjects != 109                     # subject 109 has 48 windows only
        data, chunk = data.subset(keep), chunk[keep]
        subs = np.array(sorted(set(data.subjects.tolist())))   # 8 subjects
        test_ids = subs[[(2 * fold) % 8, (2 * fold + 1) % 8]]
        rest = [s for s in subs if s not in test_ids]
        val_ids = [rest[fold % len(rest)]]
        tr_mask = ~np.isin(data.subjects, np.concatenate([test_ids, val_ids]))
        train_d = data.subset(tr_mask)
        mean = train_d.X.mean(axis=(0, 2), keepdims=True)
        std = train_d.X.std(axis=(0, 2), keepdims=True) + 1e-8
        norm = lambda d: uci_har.HARData(((d.X - mean) / std).astype(np.float32), d.y, d.subjects)
        clients, cid = {}, 0
        trc = chunk[tr_mask]
        for s in np.unique(train_d.subjects):
            for c in range(4):
                m = (train_d.subjects == s) & (trc == c)
                if m.sum() >= 30:
                    clients[cid] = _tensors(norm(train_d.subset(m))); cid += 1
        val = norm(data.subset(np.isin(data.subjects, val_ids)))
        test = norm(data.subset(np.isin(data.subjects, test_ids)))
        return Task(name, clients, _tensors(val), _tensors(test), 12, 18, 128)
    raise ValueError(name)
