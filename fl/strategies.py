"""Client-selection / configuration strategies: baselines and EcoFed (+ ablations)."""
from __future__ import annotations

import itertools

import numpy as np

BITS = (8, 32)
TAUS = (1, 2)


class Base:
    name = "base"
    prox_mu = 0.0

    def setup(self, task, em, fleet, cfg, rng):
        self.em, self.fleet, self.cfg, self.rng = em, fleet, cfg, rng
        self.L, self.m, self.N = em.L, cfg.m, len(fleet.ids)
        self.full = (self.L, 32, cfg.tau_max)

    # energy of a configuration for client k
    def e(self, k, c):
        d, b, tau = c
        return self.em.e_round(self.fleet.dev[k], d, b, tau, self.fleet.n[k], self.cfg.radio_scale)

    def feasible(self, k, c):
        return self.fleet.battery[k] - self.e(k, c) >= 0

    def observe(self, *a): pass
    def end_round(self, *a): pass

    def stat_util(self, k):
        loss = self.fleet.last_loss.get(k)
        n = self.fleet.n[k]
        return n * (loss if loss is not None else 3.0)     # optimistic for unexplored clients


class FedAvg(Base):
    name = "fedavg"

    def config(self, k): return self.full

    def eligible(self, t, k): return self.feasible(k, self.config(k))

    def select(self, t, fleet):
        el = [k for k in fleet.ids if self.eligible(t, k)]
        pick = self.rng.choice(el, size=min(self.m, len(el)), replace=False) if el else []
        return [(int(k), self.config(k)) for k in pick]


class FedProx(FedAvg):
    name = "fedprox"
    prox_mu = 0.01

    def __init__(self, mu=0.01): self.prox_mu = mu


class QSGD8(FedAvg):
    name = "fedavg_q8"

    def config(self, k): return (self.L, 8, self.cfg.tau_max)


class Paced(FedAvg):
    """FedAvg that only admits a client when its cumulative energy stays within budget*(t+1) + one burst."""
    name = "fedavg_paced"

    def eligible(self, t, k):
        e = self.e(k, self.full)
        return self.feasible(k, self.full) and \
            self.fleet.cum_e[k] + e <= self.fleet.budget[k] * (t + 1) + e


class StaticDepth(FedAvg):
    """DepthFL-style: each client gets the deepest fixed depth that fits its budget at the nominal rate."""
    name = "static_depth"

    def setup(self, *a):
        super().setup(*a)
        rate = self.m / self.N
        self.depth = {}
        for k in self.fleet.ids:
            fit = [d for d in range(1, self.L + 1)
                   if rate * self.e(k, (d, 32, self.cfg.tau_max)) <= self.fleet.budget[k]]
            self.depth[k] = max(fit) if fit else 1

    def config(self, k): return (self.depth[k], 32, self.cfg.tau_max)


class Oort(FedAvg):
    name = "oort"

    def __init__(self, eps=0.1, xi=1.0): self.eps, self.xi = eps, xi

    def setup(self, *a):
        super().setup(*a)
        self.lat = {k: self.em.macs(self.L) * self.fleet.dev[k].alpha for k in self.fleet.ids}
        self.pref = float(np.median(list(self.lat.values())))

    def select(self, t, fleet):
        el = [k for k in fleet.ids if self.eligible(t, k)]
        if not el:
            return []
        n_exp = int(round(self.eps * self.m))
        score = {k: self.stat_util(k) * (min(1.0, self.pref / self.lat[k]) ** self.xi) for k in el}
        ranked = sorted(el, key=lambda k: -score[k])
        top = ranked[:max(0, self.m - n_exp)]
        rest = [k for k in el if k not in top]
        extra = list(self.rng.choice(rest, size=min(self.m - len(top), len(rest)), replace=False)) if rest else []
        return [(int(k), self.config(k)) for k in top + extra]


class EnergyGreedy(Oort):
    """Rank by statistical utility per Joule (full configuration)."""
    name = "energy_greedy"

    def select(self, t, fleet):
        el = [k for k in fleet.ids if self.eligible(t, k)]
        if not el:
            return []
        score = {k: self.stat_util(k) / self.e(k, self.full) for k in el}
        n_exp = int(round(self.eps * self.m))
        ranked = sorted(el, key=lambda k: -score[k])
        top = ranked[:max(0, self.m - n_exp)]
        rest = [k for k in el if k not in top]
        extra = list(self.rng.choice(rest, size=min(self.m - len(top), len(rest)), replace=False)) if rest else []
        return [(int(k), self.config(k)) for k in top + extra]


class EcoFed(Base):
    """Drift-plus-penalty controller with energy queues q_k and block-coverage queues Z_j.

    score_k(c) = V * U_k(c) - q_k * e_k(c)/budget_k + (1/m) * sum_{j<=d} Z_j
    Each client takes its best configuration; the top-m positive scores are selected.
    """
    name = "ecofed"

    def __init__(self, V=2.0, rho_min=0.5, use_energy_q=True, use_cov_q=True,
                 depths=None, bits=BITS, taus=TAUS, tag=None, kappa=1.0, gamma_pow=1.0):
        self.V, self.rho_min = V, rho_min
        self.use_energy_q, self.use_cov_q = use_energy_q, use_cov_q
        self.depths_opt, self.bits_opt, self.taus_opt = depths, bits, taus
        self.kappa, self.gamma_pow = kappa, gamma_pow
        if tag:
            self.name = tag

    def setup(self, *a):
        super().setup(*a)
        self.depths = self.depths_opt or tuple(range(1, self.L + 1))
        self.configs = list(itertools.product(self.depths, self.bits_opt, self.taus_opt))
        self.Z = np.zeros(self.L)
        self.target = np.array([1.0] + [self.rho_min] * (self.L - 1))
        self.q = {k: 0.0 for k in self.fleet.ids}
        self.e_tab = {k: {c: self.e(k, c) for c in self.configs} for k in self.fleet.ids}

    def utility_cfg(self, c, u_norm):
        d, b, tau = c
        gamma = (d / self.L) ** self.gamma_pow
        return u_norm * gamma * (1 - self.kappa / (2 ** b - 1)) * np.sqrt(tau / self.cfg.tau_max)

    def select(self, t, fleet):
        us = {k: self.stat_util(k) for k in fleet.ids}
        mean_u = np.mean(list(us.values()))
        cands = []
        for k in fleet.ids:
            best, best_s = None, -np.inf
            for c in self.configs:
                e = self.e_tab[k][c]
                if fleet.battery[k] - e < 0:
                    continue
                s = self.V * self.utility_cfg(c, us[k] / mean_u)
                if self.use_energy_q:
                    s -= self.q[k] * e / fleet.budget[k]
                if self.use_cov_q:
                    s += self.Z[:c[0]].sum() / self.m
                if s > best_s:
                    best, best_s = c, s
            if best is not None and best_s > 0:
                cands.append((best_s, k, best))
        cands.sort(key=lambda x: -x[0])
        return [(int(k), c) for _, k, c in cands[:self.m]]

    def end_round(self, t, fleet, picks, updates, rho):
        sel = {k: c for k, c in picks}
        for k in fleet.ids:
            e = self.e_tab[k][sel[k]] if (k in sel and fleet.battery[k] >= 0) else 0.0
            took = k in sel
            e_real = e if took else 0.0
            self.q[k] = max(self.q[k] + e_real / fleet.budget[k] - 1.0, 0.0)
        self.Z = np.maximum(self.Z + self.target - rho, 0.0) if updates else self.Z + self.target * 0.0
        self.Z[0] = 0.0


class ForcedDepth(FedAvg):
    """Theory probe: uniform client sampling; depth L w.p. rho, otherwise depth 1 (so Pr[d>=j]=rho, j>=2)."""
    name = "forced_depth"

    def __init__(self, rho=1.0): self.rho = rho

    def select(self, t, fleet):
        pick = self.rng.choice(fleet.ids, size=min(self.m, len(fleet.ids)), replace=False)
        return [(int(k), (self.L if self.rng.random() < self.rho else 1, 32, self.cfg.tau_max))
                for k in pick]


def registry(name: str, **kw):
    table = {
        "fedavg": FedAvg, "fedprox": FedProx, "fedavg_q8": QSGD8, "fedavg_paced": Paced,
        "static_depth": StaticDepth, "oort": Oort, "energy_greedy": EnergyGreedy,
        "forced_depth": ForcedDepth, "ecofed": EcoFed,
        "ecofed_noQ": lambda **k: EcoFed(use_energy_q=False, tag="ecofed_noQ", **k),
        "ecofed_noZ": lambda **k: EcoFed(use_cov_q=False, tag="ecofed_noZ", **k),
        "ecofed_fixdepth": lambda **k: EcoFed(depths=(4,), tag="ecofed_fixdepth", **k),
        "ecofed_fixprec": lambda **k: EcoFed(bits=(32,), tag="ecofed_fixprec", **k),
        "ecofed_fixtau": lambda **k: EcoFed(taus=(2,), tag="ecofed_fixtau", **k),
    }
    return table[name](**kw)
