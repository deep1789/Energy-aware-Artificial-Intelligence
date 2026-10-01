"""Federated simulation engine for heterogeneous-depth multi-exit training with energy accounting."""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from energy.model import DEVICE_MIX, DEVICE_PROFILES, EnergyModel
from fl.tasks import Task
from models.multiexit import MultiExitCNN1D, multi_exit_loss

torch.set_num_threads(1)
FULL = None  # placeholder, set per run: (L, 32, tau_max)


# ---------------------------------------------------------------- quantisation / training
def stochastic_quantize(t: torch.Tensor, bits: int, gen: torch.Generator) -> torch.Tensor:
    if bits >= 32:
        return t
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-12:
        return t
    levels = 2 ** bits - 1
    x = (t - lo) / (hi - lo) * levels
    fl = x.floor()
    x = fl + (torch.rand(t.shape, generator=gen) < (x - fl)).float()
    return x / levels * (hi - lo) + lo


def local_train(model: MultiExitCNN1D, gstate: dict, data, depth: int, tau: int, bits: int,
                lr: float, bs: int, gen: torch.Generator, prox_mu: float = 0.0):
    """Train blocks/heads 1..depth for tau epochs; return (delta over those blocks, mean loss)."""
    model.load_state_dict(gstate)
    model.train()
    X, y = data
    n = len(y)
    params = [p for j in range(depth) for p in itertools.chain(model.blocks[j].parameters(),
                                                               model.heads[j].parameters())]
    opt = torch.optim.SGD(params, lr=lr)
    gref = [p.detach().clone() for p in params] if prox_mu > 0 else None
    first_epoch_loss, cnt = 0.0, 0
    for ep in range(tau):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits = model(X[idx], depth)
            loss = multi_exit_loss(logits, y[idx])
            if ep == 0:
                first_epoch_loss += loss.item() * len(idx); cnt += len(idx)
            if gref is not None:
                loss = loss + 0.5 * prox_mu * sum(((p - g) ** 2).sum() for p, g in zip(params, gref))
            loss.backward()
            opt.step()
    new = model.state_dict()
    delta = {}
    for j in range(depth):
        for k in model.block_param_names(j):
            delta[k] = stochastic_quantize(new[k] - gstate[k], bits, gen)
    return delta, first_epoch_loss / max(cnt, 1)


def aggregate(model, gstate, updates):
    """Block-wise weighted average of deltas from clients whose depth covers the block."""
    new = {k: v.clone() for k, v in gstate.items()}
    for j in range(model.n_exits):
        ups = [(n, d) for n, dep, d in updates if dep > j]
        if not ups:
            continue
        w = sum(n for n, _ in ups)
        for k in model.block_param_names(j):
            new[k] = gstate[k] + sum(n * d[k] for n, d in ups) / w
    return new


@torch.no_grad()
def predict_probs(model, gstate, X, bs=1024):
    model.load_state_dict(gstate)
    model.eval()
    outs = [[] for _ in range(model.n_exits)]
    for i in range(0, len(X), bs):
        for j, z in enumerate(model(X[i:i + bs])):
            outs[j].append(F.softmax(z, dim=1))
    return torch.stack([torch.cat(o) for o in outs], dim=1)     # (n, L, C)


def full_grad_norm_sq(model, gstate, clients, max_n=4000, seed=0):
    """||grad F(w)||^2 for the sample-weighted multi-exit objective over all clients (full depth)."""
    model.load_state_dict(gstate)
    model.train()
    Xs = torch.cat([c[0] for c in clients.values()])
    ys = torch.cat([c[1] for c in clients.values()])
    if len(ys) > max_n:
        idx = torch.randperm(len(ys), generator=torch.Generator().manual_seed(seed))[:max_n]
        Xs, ys = Xs[idx], ys[idx]
    model.zero_grad()
    for i in range(0, len(ys), 512):
        loss = multi_exit_loss(model(Xs[i:i + 512]), ys[i:i + 512]) * (len(ys[i:i + 512]) / len(ys))
        loss.backward()
    return float(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))


# ---------------------------------------------------------------- fleet
@dataclass
class Fleet:
    ids: list
    n: dict
    dev: dict
    budget: dict          # per-round average budget (J)
    battery: dict
    cum_e: dict
    last_loss: dict = field(default_factory=dict)
    q: dict = field(default_factory=dict)     # energy virtual queues (budget units)


def make_fleet(task: Task, em: EnergyModel, m: int, phi: float, T: int, seed: int,
               tau_max: int = 2, mix=DEVICE_MIX) -> Fleet:
    rng = np.random.default_rng(seed)
    ids = list(task.clients)
    N = len(ids)
    counts = [int(round(f * N)) for _, f in mix]
    counts[-1] = N - sum(counts[:-1])
    names = np.repeat([nm for nm, _ in mix], counts)
    rng.shuffle(names)
    n = {i: len(task.clients[i][1]) for i in ids}
    n_ref = float(np.mean(list(n.values())))
    mid = DEVICE_PROFILES["mid"]
    e_ref = (m / N) * em.e_round(mid, em.L, 32, tau_max, n_ref)
    dev = {i: DEVICE_PROFILES[nm] for i, nm in zip(ids, names)}
    budget = {i: phi * e_ref * dev[i].battery_scale for i in ids}
    battery = {i: budget[i] * T for i in ids}
    return Fleet(ids, n, dev, budget, battery, {i: 0.0 for i in ids},
                 q={i: 0.0 for i in ids})


# ---------------------------------------------------------------- run loop
@dataclass
class RunCfg:
    T: int = 80
    m: int = 5
    phi: float = 0.4
    lr: float = 0.05
    bs: int = 32
    eval_every: int = 4
    tau_max: int = 2
    radio_scale: float = 1.0
    seed: int = 0


def run_fl(task: Task, strategy, cfg: RunCfg, record_final: bool = True):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(1000 + cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MultiExitCNN1D(task.in_ch, task.n_classes, task.in_len)
    em = EnergyModel(model.block_stats(), task.in_ch, task.in_len)
    fleet = make_fleet(task, em, cfg.m, cfg.phi, cfg.T, cfg.seed, cfg.tau_max)
    gstate = copy.deepcopy(model.state_dict())
    strategy.setup(task, em, fleet, cfg, rng)
    L = model.n_exits
    log = dict(round=[], cum_energy=[], acc=[], val_acc=[], alive=[], participants=[], rho=[], depth_mean=[])
    cum_energy, t0 = 0.0, time.time()
    rho_hist = []
    for t in range(cfg.T):
        picks = strategy.select(t, fleet)                         # [(cid, (d, bits, tau))]
        updates, rho_num, rho_den = [], np.zeros(L), 0.0
        round_e = 0.0
        for cid, (d, bits, tau) in picks:
            e = em.e_round(fleet.dev[cid], d, bits, tau, fleet.n[cid], cfg.radio_scale)
            if fleet.battery[cid] - e < 0:
                continue                                           # infeasible: client drops
            delta, loss = local_train(model, gstate, task.clients[cid], d, tau, bits, cfg.lr,
                                      cfg.bs, gen, getattr(strategy, "prox_mu", 0.0))
            fleet.battery[cid] -= e; fleet.cum_e[cid] += e; round_e += e
            fleet.last_loss[cid] = loss
            updates.append((fleet.n[cid], d, delta))
            rho_num += fleet.n[cid] * (np.arange(1, L + 1) <= d); rho_den += fleet.n[cid]
            strategy.observe(cid, d, bits, tau, e, loss)
        if updates:
            gstate = aggregate(model, gstate, updates)
        cum_energy += round_e
        rho = rho_num / rho_den if rho_den > 0 else np.zeros(L)
        strategy.end_round(t, fleet, picks, updates, rho)
        rho_hist.append(rho)
        if (t + 1) % cfg.eval_every == 0 or t == cfg.T - 1:
            P = predict_probs(model, gstate, task.test[0])
            acc = (P.argmax(-1) == task.test[1][:, None]).float().mean(0).tolist()
            alive = sum(fleet.battery[i] >= 0.1 * fleet.budget[i] for i in fleet.ids)
            Pv = predict_probs(model, gstate, task.val[0])
            log["val_acc"].append((Pv.argmax(-1) == task.val[1][:, None]).float().mean(0).tolist())
            log["round"].append(t + 1); log["cum_energy"].append(cum_energy)
            log["acc"].append(acc); log["alive"].append(int(alive))
            log["participants"].append(len(updates))
            log["rho"].append(np.mean(rho_hist[-cfg.eval_every:], axis=0).tolist())
            log["depth_mean"].append(float(np.mean([d for _, d, _ in updates])) if updates else 0.0)
    out = dict(log=log, total_energy=cum_energy, secs=time.time() - t0,
               fleet_capacity=float(sum(fleet.budget[i] * cfg.T for i in fleet.ids)),
               per_client_energy=[fleet.cum_e[i] for i in fleet.ids],
               per_client_capacity=[fleet.budget[i] * cfg.T for i in fleet.ids],
               final_battery=[fleet.battery[i] for i in fleet.ids])
    if record_final:
        out["val_probs"] = predict_probs(model, gstate, task.val[0]).numpy().astype(np.float16)
        out["test_probs"] = predict_probs(model, gstate, task.test[0]).numpy().astype(np.float16)
        out["gradnorm_sq"] = full_grad_norm_sq(model, gstate, task.clients, seed=cfg.seed)
    return out
