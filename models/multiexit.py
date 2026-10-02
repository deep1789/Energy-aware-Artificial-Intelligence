"""Multi-exit 1D-CNN for windowed inertial data.

Block l = Conv1d -> GroupNorm -> ReLU -> MaxPool(2). Exit head l = global average pool + Linear.
Exit j uses blocks 1..j, so a client of depth d only needs blocks 1..d and heads 1..d.
GroupNorm (not BatchNorm) keeps federated averaging well defined.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockStats:
    """Analytic per-sample cost of one block + its exit head (forward pass)."""
    macs: int          # multiply-accumulates, block + head
    act_elems: int     # output activation elements of the block
    params: int        # parameters of block + head (what a client must download/upload)
    head_macs: int


class MultiExitCNN1D(nn.Module):
    def __init__(self, in_ch: int = 9, n_classes: int = 6, in_len: int = 128,
                 widths: tuple[int, ...] = (16, 32, 48, 64), kernel: int = 5):
        super().__init__()
        self.widths, self.n_exits = widths, len(widths)
        self.in_ch, self.in_len, self.n_classes, self.kernel = in_ch, in_len, n_classes, kernel
        blocks, heads, c_prev = [], [], in_ch
        for w in widths:
            blocks.append(nn.Sequential(
                nn.Conv1d(c_prev, w, kernel, padding=kernel // 2),
                nn.GroupNorm(4, w), nn.ReLU(inplace=True), nn.MaxPool1d(2)))
            heads.append(nn.Linear(w, n_classes))
            c_prev = w
        self.blocks, self.heads = nn.ModuleList(blocks), nn.ModuleList(heads)

    def forward(self, x: torch.Tensor, depth: int | None = None) -> list[torch.Tensor]:
        """Logits of exits 1..depth (all exits when depth is None)."""
        depth = depth or self.n_exits
        outs = []
        for j in range(depth):
            x = self.blocks[j](x)
            outs.append(self.heads[j](x.mean(dim=2)))
        return outs

    def block_stats(self) -> list[BlockStats]:
        stats, c_prev, length = [], self.in_ch, self.in_len
        for j, w in enumerate(self.widths):
            conv_macs = c_prev * w * self.kernel * length
            head_macs = w * self.n_classes
            params = (sum(p.numel() for p in self.blocks[j].parameters())
                      + sum(p.numel() for p in self.heads[j].parameters()))
            length //= 2
            stats.append(BlockStats(conv_macs + head_macs, w * length, params, head_macs))
            c_prev = w
        return stats

    def block_param_names(self, j: int) -> list[str]:
        """State-dict keys that belong to block j and head j (0-based)."""
        return [k for k in self.state_dict()
                if k.startswith(f"blocks.{j}.") or k.startswith(f"heads.{j}.")]


def exit_weights(n: int, scheme: str = "uniform") -> list[float]:
    """Normalised weights over the first n exits. 'linear' puts weight proportional to the exit index."""
    raw = {"uniform": [1.0] * n, "linear": [float(j + 1) for j in range(n)],
           "quad": [float((j + 1) ** 2) for j in range(n)]}[scheme]
    z = sum(raw)
    return [r / z for r in raw]


def multi_exit_loss(logits: list[torch.Tensor], y: torch.Tensor, scheme: str = "uniform") -> torch.Tensor:
    """Weighted cross-entropy over the exits that were computed (weights normalised over those exits)."""
    w = exit_weights(len(logits), scheme)
    return sum(wi * F.cross_entropy(z, y) for wi, z in zip(w, logits))
