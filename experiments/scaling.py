"""Controller overhead vs fleet size N (synthetic fleets; no training). Run on an idle machine."""
import json, time
import numpy as np
from energy.model import DEVICE_MIX, DEVICE_PROFILES, EnergyModel
from fl.sim import Fleet, RunCfg
from fl.strategies import EcoFed, Oort
from models.multiexit import MultiExitCNN1D


def synthetic_fleet(N, em, m, phi, T, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(range(N))
    counts = [int(round(f * N)) for _, f in DEVICE_MIX]; counts[-1] = N - sum(counts[:-1])
    names = np.repeat([n for n, _ in DEVICE_MIX], counts); rng.shuffle(names)
    n = {i: int(rng.integers(250, 550)) for i in ids}
    dev = {i: DEVICE_PROFILES[nm] for i, nm in zip(ids, names)}
    e_ref = (m / N) * em.e_round(DEVICE_PROFILES["mid"], em.L, 32, 2, float(np.mean(list(n.values()))))
    budget = {i: phi * e_ref * dev[i].battery_scale for i in ids}
    fl = Fleet(ids, n, dev, budget, {i: budget[i] * T for i in ids}, {i: 0.0 for i in ids}, q={i: 0.0 for i in ids})
    fl.last_loss = {i: float(rng.uniform(0.2, 2.0)) for i in ids}
    return fl


if __name__ == "__main__":
    model = MultiExitCNN1D(9, 6); em = EnergyModel(model.block_stats(), 9, 128)
    out = []
    for N in (100, 1000, 10000, 100000):
        m = max(5, N // 20)
        fleet = synthetic_fleet(N, em, m, 0.25, 80)
        cfg = RunCfg(m=m)
        for cls, name in ((EcoFed, "ecofed"), (Oort, "oort")):
            st = cls(); st.setup(None, em, fleet, cfg, np.random.default_rng(0))
            ts = []
            for t in range(3):
                t0 = time.perf_counter(); picks = st.select(t, fleet); ts.append(time.perf_counter() - t0)
            out.append(dict(N=N, method=name, seconds=float(np.median(ts)), picked=len(picks)))
            print(out[-1], flush=True)
    json.dump(out, open("results/scaling.json", "w"), indent=1)
