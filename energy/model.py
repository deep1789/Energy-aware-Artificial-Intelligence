"""Analytic, device-aware energy model for federated multi-exit training and inference.

IMPORTANT: no power meter is available in the development environment, so every
device constant below is an *assumption* taken from typical published orders of magnitude
(see DEVICE_PROFILES). Results are therefore model-based energy, and the experiments include
sensitivity sweeps over these constants. Hardware calibration is left to future work.

E_comp(d, tau) = tau * n * [alpha * MAC_{1:d} + beta * Bytes_{1:d}]
E_comm(d, b)   = P_tx * bits_up(d, b) / R + P_rx * bits_down(d) / R   (expressed as J/bit)
"""
from __future__ import annotations

from dataclasses import dataclass

from models.multiexit import BlockStats

BYTES_PER_PARAM = 4
TRAIN_MAC_FACTOR = 3.0      # forward + backward ~ 3x forward MACs
BATCH = 32


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    alpha: float        # J per MAC (includes static power amortised over compute)
    beta: float         # J per byte of memory traffic
    tx_j_per_bit: float
    rx_j_per_bit: float
    battery_scale: float  # relative energy budget (battery) of the device class


# Orders of magnitude: Pi-class ~4 W at ~2 GMAC/s; Jetson-Nano-class ~7 W at ~10 GMAC/s;
# Orin-class ~10 W at ~50 GMAC/s. Wi-Fi ~50 nJ/bit, LTE ~200 nJ/bit.
DEVICE_PROFILES = {
    "weak":   DeviceProfile("weak",   2.0e-9, 5.0e-10, 5.0e-8, 2.5e-8, 0.5),
    "mid":    DeviceProfile("mid",    0.7e-9, 3.0e-10, 5.0e-8, 2.5e-8, 1.0),
    "strong": DeviceProfile("strong", 0.2e-9, 1.5e-10, 2.0e-7, 1.0e-7, 2.0),
}
DEVICE_MIX = (("weak", 0.4), ("mid", 0.4), ("strong", 0.2))


class EnergyModel:
    def __init__(self, stats: list[BlockStats], in_ch: int, in_len: int):
        self.stats, self.L = stats, len(stats)
        self.in_elems = in_ch * in_len

    # ---- per-sample compute quantities (depth d in 1..L) ----
    def macs(self, d: int) -> float:
        return sum(s.macs for s in self.stats[:d])

    def bytes_per_sample(self, d: int) -> float:
        act = self.in_elems + sum(s.act_elems for s in self.stats[:d])
        weights = sum(s.params for s in self.stats[:d])
        # activations: written fwd, read bwd, grad written/read -> ~4 passes;
        # weights: read fwd, read bwd, read+write update, amortised over the batch.
        return BYTES_PER_PARAM * (4 * act + 4 * weights / BATCH)

    def params(self, d: int) -> int:
        return sum(s.params for s in self.stats[:d])

    # ---- energies ----
    def e_comp(self, dev: DeviceProfile, d: int, tau: int, n: int) -> float:
        per = dev.alpha * TRAIN_MAC_FACTOR * self.macs(d) + dev.beta * self.bytes_per_sample(d)
        return tau * n * per

    def e_comm(self, dev: DeviceProfile, d: int, bits: int, radio_scale: float = 1.0) -> float:
        n_par = self.params(d)
        up = dev.tx_j_per_bit * n_par * bits
        down = dev.rx_j_per_bit * n_par * 32
        return radio_scale * (up + down)

    def e_round(self, dev, d, bits, tau, n, radio_scale=1.0) -> float:
        return self.e_comp(dev, d, tau, n) + self.e_comm(dev, d, bits, radio_scale)

    # ---- inference ----
    def e_infer_exit(self, dev: DeviceProfile, j: int) -> float:
        """Energy of one inference that runs blocks 1..j and head j (j in 1..L)."""
        macs = self.macs(j)
        byt = BYTES_PER_PARAM * (self.in_elems + sum(s.act_elems for s in self.stats[:j]))
        return dev.alpha * macs + dev.beta * byt
