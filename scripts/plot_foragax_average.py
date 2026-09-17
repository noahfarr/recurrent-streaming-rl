import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from styles import FIG_WIDTH, GOLDEN_RATIO, apply as apply_style

apply_style()

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/average_reward.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/.average.pkl")
ENV_ID = "ForagaxNeverEndingRelearning-v1"
SINCE = "2026-09-05T00:00:00"
TOTAL_TIMESTEPS = 5_000_000
PERIOD = 500_000
METRIC = "average_reward"
PPO_ENTROPY = 0.1
ALGORITHMS = ["stream_q", "intentional_q", "real_time_ppo"]
TITLES = {"stream_q": "stream RQ($\\lambda$)", "intentional_q": "intentional RQ($\\lambda$)",
          "real_time_ppo": "PPO"}
COLOURS = dict(zip(ALGORITHMS, sns.color_palette("Set1", len(ALGORITHMS))))


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={
        "state": "finished",
        "config.total_timesteps": TOTAL_TIMESTEPS,
        "config.environment.env_id": ENV_ID,
        "createdAt": {"$gt": SINCE},
    }, order="-created_at")
    records = defaultdict(list)
    for run in runs:
        groups = run.config.get("groups") or {}
        kwargs = (run.config.get("environment") or {}).get("kwargs") or {}
        if kwargs.get("observation_type") != "object":
            continue
        algorithm, cell = groups.get("algorithm"), groups.get("cell")
        if algorithm not in ALGORITHMS or cell != "rtu":
            continue
        if algorithm == "real_time_ppo" and (run.config.get("algorithm") or {}).get(
            "entropy_coefficient"
        ) != PPO_ENTROPY:
            continue
        steps, values = [], []
        for row in run.history(keys=[METRIC], samples=200, pandas=False):
            if row.get(METRIC) is not None:
                steps.append(row["_step"])
                values.append(row[METRIC])
        if steps:
            records[algorithm].append((np.array(steps), np.array(values)))
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    low, high = np.percentile(values, [25, 75], axis=0)
    return np.nanmean(np.where((values >= low) & (values <= high), values, np.nan), axis=0)


def main():
    records = fetch()
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, axis = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * 0.62, FIG_WIDTH * 0.62 / GOLDEN_RATIO),
        constrained_layout=True,
    )
    for boundary in range(PERIOD, TOTAL_TIMESTEPS, PERIOD):
        axis.axvline(boundary / 1e6, color="0.5", linewidth=0.5, linestyle=":", alpha=0.5)
    summary = {}
    for algorithm in ALGORITHMS:
        curves = records.get(algorithm, [])
        if not curves:
            continue
        start = max(s.min() for s, _ in curves)
        limit = min(s.max() for s, _ in curves)
        grid = np.linspace(start, limit, 200)
        stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
        centre = interquartile_mean(stacked)
        spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
        axis.plot(grid / 1e6, centre, color=COLOURS[algorithm], linewidth=1.0)
        axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                          color=COLOURS[algorithm], alpha=0.2, linewidth=0)
        tail = grid >= start + 0.9 * (limit - start)
        per_seed = stacked[:, tail].mean(axis=1)
        summary[algorithm] = (per_seed.mean(), per_seed.std() / np.sqrt(len(per_seed)),
                              len(per_seed))
    axis.set_xlabel("Frames (millions)", fontsize=label_size)
    axis.set_ylabel("Average Reward", fontsize=label_size)
    axis.set_xticks([0, 1, 2, 3, 4, 5])
    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=tick_size)
    for spine in axis.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
    handles = [plt.Line2D([], [], color=COLOURS[a], linewidth=1.0, label=TITLES[a])
               for a in ALGORITHMS if a in summary]
    figure.legend(handles=handles, loc="outside upper center", frameon=False,
                  fontsize=tick_size, ncol=len(handles), handlelength=1.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}\n")
    print("Last 10% Average Reward (their summary statistic), RTU:")
    for algorithm, (mean, sem, n) in summary.items():
        print(f"  {TITLES[algorithm]:>26}  {mean:6.3f} +- {sem:5.3f}   n={n}")


if __name__ == "__main__":
    main()
