import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
import submitit
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.utils.carbs import build_carbs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("sweep")

project_root = str(Path(__file__).resolve().parent.parent)
POLL_SECONDS = 30.0


def launch(overrides, run_dir):
    cmd = [sys.executable, "main.py", *overrides, f"hydra.run.dir={run_dir}"]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    }
    subprocess.run(cmd, cwd=project_root, env=env, check=True)
    return json.loads((Path(run_dir) / "result.json").read_text())


def run_trial(overrides, run_dir):
    return launch(overrides, run_dir)


def record_observation(path, hyperparameters, score, cost, is_failure):
    record = {
        "hyperparameters": hyperparameters,
        "score": score,
        "cost": cost,
        "is_failure": is_failure,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def save_results(output_dir, best_value, best_params):
    OmegaConf.save(
        OmegaConf.create({"best_value": best_value, "best_params": best_params}),
        output_dir / "optimization_results.yaml",
    )


@hydra.main(version_base=None, config_path="../config", config_name="sweep")
def main(cfg):
    sweep = cfg.sweep
    optimizer, metric, cost_key = build_carbs(cfg)

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    base_overrides = [
        override
        for override in HydraConfig.get().overrides.task
        if not override.startswith("sweep.") and not override.startswith("hydra.")
    ] + [
        "logger=wandb",
        f"+sweep={output_dir.name}",
        f"+hyperparameters=[{','.join(sweep.params.keys())}]",
    ]

    num_trials = math.inf if sweep.num_trials is None else int(sweep.num_trials)
    num_jobs = int(sweep.num_jobs)

    executor = submitit.AutoExecutor(
        folder=str(output_dir / "submitit"), cluster=sweep.cluster
    )
    executor.update_parameters(
        name="trial",
        nodes=1,
        tasks_per_node=1,
        **OmegaConf.to_container(sweep.executor, resolve=True),
    )

    checkpoint = output_dir / "observations.jsonl"
    inflight = {}
    launched = completed = 0
    best_value = best_params = None
    groups = {}

    while completed < num_trials:
        while len(inflight) < num_jobs and launched < num_trials:
            suggestion_id, hyperparameters = optimizer.suggest()
            run_dir = output_dir / f"trial_{launched:04d}"
            overrides = (
                base_overrides
                + [f"{k}={v}" for k, v in hyperparameters.items()]
                + [f"seed={launched}"]
            )
            job = executor.submit(run_trial, overrides, str(run_dir))
            inflight[job.job_id] = (
                job,
                suggestion_id,
                hyperparameters,
                time.monotonic(),
            )
            launched += 1

        done = [jid for jid, (job, *_) in inflight.items() if job.done()]
        if not done:
            time.sleep(POLL_SECONDS)
            continue

        for jid in done:
            job, suggestion_id, hyperparameters, start_time = inflight.pop(jid)
            try:
                result = job.result()
                cost = float(result[cost_key])
                score = float(result[metric])
                optimizer.observe(suggestion_id, hyperparameters, score, cost)
                record_observation(checkpoint, hyperparameters, score, cost, False)
                key = json.dumps(hyperparameters, sort_keys=True, default=float)
                group = groups.setdefault(
                    key, {"params": hyperparameters, "scores": []}
                )
                group["scores"].append(score)
                best = max(
                    groups.values(),
                    key=lambda g: sum(g["scores"]) / len(g["scores"]),
                )
                best_value = sum(best["scores"]) / len(best["scores"])
                best_params = best["params"]
                save_results(output_dir, best_value, best_params)
            except Exception:
                cost = time.monotonic() - start_time
                optimizer.observe(
                    suggestion_id, hyperparameters, 0.0, cost, is_failure=True
                )
                record_observation(checkpoint, hyperparameters, 0.0, cost, True)
            completed += 1

    save_results(output_dir, best_value, best_params)


if __name__ == "__main__":
    main()
