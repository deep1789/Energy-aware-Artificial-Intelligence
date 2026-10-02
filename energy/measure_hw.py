"""Metered-hardware measurement harness (NOT executed in the paper: no power meter was available).

Measures the energy of local training on the device it runs on, for every depth and batch size, and writes the CSV that
energy/fit_profile.py consumes. Three power sources are supported; only 'csv' is exercised by the unit tests:

  --source csv       an external meter (Joulescope, Nordic PPK2, USB power meter) logs "unix_seconds,watts" to a file
                     while this script runs; the script reads the log afterwards (clocks must be synchronised).
  --source tegrastats Jetson: runs `tegrastats --interval 100` and parses the VDD_IN (total board power) field, in mW.
  --source ina219    Raspberry Pi with an INA219 on the supply line (needs the `pi-ina219` package), sampled at ~50 Hz.

Protocol (per device): 1. idle for 60 s -> idle power. 2. for depth in 1..4 and batch in (16, 32, 64): warm up 3 s, run
training steps for --seconds s, record start/end timestamps. 3. energy above idle = integral of (P - P_idle) over the
window. 4. fit with energy/fit_profile.py. 5. radio: transfer a file of known size over the real link, repeat for upload
and download, energy above idle / (8 * bytes). Pin CPU governor/frequency, disable thermal throttling effects by logging
temperature, run >= 5 repetitions and report the spread.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import threading
import time

import numpy as np
import torch

from models.multiexit import MultiExitCNN1D, multi_exit_loss


class CsvPower:
    def __init__(self, path):
        d = np.loadtxt(path, delimiter=",")
        self.t, self.w = d[:, 0], d[:, 1]

    def idle_power(self, t0, t1):
        m = (self.t >= t0) & (self.t <= t1)
        return float(self.w[m].mean())

    def energy(self, t0, t1, idle_w):
        m = (self.t >= t0) & (self.t <= t1)
        t, w = self.t[m], self.w[m] - idle_w
        return float(np.trapezoid(w, t)) if len(t) > 1 else 0.0


class SampledPower(CsvPower):
    """Base for sources sampled in a background thread."""
    def __init__(self):
        self._t, self._w, self._run = [], [], False

    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False
        self.t, self.w = np.array(self._t), np.array(self._w)


class Tegrastats(SampledPower):
    def _loop(self):
        p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        while self._run:
            line = p.stdout.readline()
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                self._t.append(time.time()); self._w.append(int(m.group(1)) / 1000.0)
        p.terminate()


class Ina219(SampledPower):
    def _loop(self):
        from ina219 import INA219
        ina = INA219(0.1, address=0x40); ina.configure()
        while self._run:
            self._t.append(time.time()); self._w.append(ina.power() / 1000.0)
            time.sleep(0.02)


def train_window(in_ch, n_classes, depth, batch, seconds):
    m = MultiExitCNN1D(in_ch, n_classes)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x, y = torch.randn(batch, in_ch, 128), torch.randint(0, n_classes, (batch,))
    end = time.time() + 3.0
    while time.time() < end:                       # warm-up
        opt.zero_grad(); multi_exit_loss(m(x, depth), y).backward(); opt.step()
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        opt.zero_grad(); multi_exit_loss(m(x, depth), y).backward(); opt.step(); steps += 1
    return t0, time.time(), steps * batch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "tegrastats", "ina219"], required=True)
    ap.add_argument("--csv", help="meter log for --source csv (read after the run)")
    ap.add_argument("--out", default="measurements.csv")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--in-ch", type=int, default=9); ap.add_argument("--classes", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(1)
    src = {"tegrastats": Tegrastats, "ina219": Ina219}.get(a.source)
    if src:
        src = src(); src.start()
    print("idle for 60 s ..."); t_idle0 = time.time(); time.sleep(60); t_idle1 = time.time()
    windows = []
    for depth in (1, 2, 3, 4):
        for batch in (16, 32, 64):
            t0, t1, n = train_window(a.in_ch, a.classes, depth, batch, a.seconds)
            windows.append((depth, batch, n, t0, t1)); print("done", depth, batch)
    if src:
        src.stop()
    else:
        src = CsvPower(a.csv)
    idle_w = src.idle_power(t_idle0, t_idle1)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["depth", "batch", "samples", "joules_above_idle"])
        for depth, batch, n, t0, t1 in windows:
            w.writerow([depth, batch, n, src.energy(t0, t1, idle_w)])
    print("idle power (W):", idle_w, "-> now run: python -m energy.fit_profile", a.out)
