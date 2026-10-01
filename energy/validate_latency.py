"""Sanity check of the *relative* cost model: wall-clock time per training sample vs modelled MACs/bytes.

This validates the depth/epoch scaling of the cost model on the CPU we have; it is a latency proxy, not an
energy measurement. Run on an otherwise idle machine."""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from models.multiexit import MultiExitCNN1D, multi_exit_loss
from energy.model import EnergyModel

torch.set_num_threads(1)

def bench(in_ch, n_classes, depth, bs=32, reps=40):
    m = MultiExitCNN1D(in_ch, n_classes)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x, y = torch.randn(bs, in_ch, 128), torch.randint(0, n_classes, (bs,))
    for _ in range(5):
        opt.zero_grad(); multi_exit_loss(m(x, depth), y).backward(); opt.step()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        opt.zero_grad(); multi_exit_loss(m(x, depth), y).backward(); opt.step()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) / bs

if __name__ == "__main__":
    out = {}
    for name, (c, k) in {"uci_har": (9, 6), "pamap2": (18, 12)}.items():
        m = MultiExitCNN1D(c, k)
        em = EnergyModel(m.block_stats(), c, 128)
        rows = []
        for d in range(1, 5):
            rows.append(dict(depth=d, sec_per_sample=bench(c, k, d), macs=3 * em.macs(d), bytes=em.bytes_per_sample(d)))
        t = np.array([r["sec_per_sample"] for r in rows]); mac = np.array([r["macs"] for r in rows])
        A = np.vstack([mac, np.ones_like(mac)]).T
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        pred = A @ coef
        r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        out[name] = dict(rows=rows, r2_time_vs_macs=float(r2), rel_cost_model=(mac / mac[0]).tolist(),
                         rel_time=(t / t[0]).tolist())
        print(name, "R2(time~MACs)=%.3f" % r2, "rel time", np.round(t / t[0], 2), "rel MACs", np.round(mac / mac[0], 2))
    json.dump(out, open("results/latency_validation.json", "w"), indent=1)
