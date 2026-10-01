"""Shared job runner: a job is (dataset, fold, strategy name + kwargs, RunCfg fields) -> result pickle."""
from __future__ import annotations

import hashlib
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

from fl.sim import RunCfg, run_fl
from fl.strategies import registry
from fl.tasks import build_task

_TASKS: dict = {}
RAW = Path("results/raw")


def job_key(job: dict) -> str:
    s = json.dumps(job, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def run_job(job: dict):
    path = RAW / job["exp"] / f"{job_key(job)}.pkl"
    if path.exists():
        return job, pickle.load(open(path, "rb"))
    ds, fold = job["dataset"], job.get("fold", 0)
    if (ds, fold) not in _TASKS:
        _TASKS[(ds, fold)] = build_task(ds, fold)
    task = _TASKS[(ds, fold)]
    cfg = RunCfg(**job.get("cfg", {}))
    strat = registry(job["method"], **job.get("kw", {}))
    res = run_fl(task, strat, cfg, record_final=job.get("record_final", True))
    res["job"] = job
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(path, "wb"))
    return job, res


def run_all(jobs: list[dict], procs: int = 4):
    with Pool(procs) as p:
        return list(p.imap_unordered(run_job, jobs))


def load_results(exp: str):
    """Yield (job, result) for every finished job of an experiment (job stored alongside)."""
    return [(r["job"], r) for r in (pickle.load(open(f, "rb")) for f in (RAW / exp).glob("*.pkl"))]
