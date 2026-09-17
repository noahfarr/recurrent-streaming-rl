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
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/trace_preconditioning.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/.precondition.pkl")
ENV_ID = "ForagaxNeverEndingRelearning-v1"
SINCE = "2026-09-05T00:00:00"
TOTAL_TIMESTEPS = 5_000_000
PERIOD = 500_000
METRIC = "reward_rate"
CELLS = ["rtu", "min_gru", "ffn"]
LABELS = {"rtu": "RTU", "min_gru": "minGRU", "ffn": "FFN"}
COLOURS = dict(zip(CELLS, sns.color_palette("Set1", len(CELLS))))
OPTIMIZERS = {"Intentional": "--", "TracePreconditionedIntentional": "-"}
OPTIMIZER_LABELS = {"Intentional": "gradient preconditioner",
                    "TracePreconditionedIntentional": "trace preconditioner"}


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={
        "state": "finished", "config.total_timesteps": TOTAL_TIMESTEPS,
        "config.environment.env_id": ENV_ID, "createdAt": {"$gt": SINCE}},
        order="-created_at")
    records = defaultdict(list)
    for run in runs:
        groups = run.config.get("groups") or {}
        kwargs = (run.config.get("environment") or {}).get("kwargs") or {}
        if kwargs.get("observation_type") != "object":
            continue
        if groups.get("algorithm") != "intentional_q" or groups.get("cell") not in CELLS:
            continue
        optimizer = str((run.config.get("q_optimizer") or {}).get("_target_", "")).split(".")[-1]
        if optimizer not in OPTIMIZERS:
            continue
        steps, values = [], []
        for row in run.history(keys=[METRIC], samples=200, pandas=False):
            if row.get(METRIC) is not None:
                steps.append(row["_step"])
                values.append(row[METRIC])
        if steps:
            records[(groups["cell"], optimizer)].append((np.array(steps), np.array(values)))
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    low, high = np.percentile(values, [25, 75], axis=0)
    return np.nanmean(np.where((values >= low) & (values <= high), values, np.nan), axis=0)


def main():
    records = fetch()
    for key in sorted(records):
        print(f"  {key}: {len(records[key])} seeds")
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, axis = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * 0.62, FIG_WIDTH * 0.62 / GOLDEN_RATIO),
        constrained_layout=True)
    for boundary in range(PERIOD, TOTAL_TIMESTEPS, PERIOD):
        axis.axvline(boundary / 1e6, color="0.5", linewidth=0.5, linestyle=":", alpha=0.5)
    for (cell, optimizer), curves in sorted(records.items()):
        start = max(s.min() for s, _ in curves)
        limit = min(s.max() for s, _ in curves)
        grid = np.linspace(start, limit, 200)
        stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
        centre = interquartile_mean(stacked)
        spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
        axis.plot(grid / 1e6, centre, OPTIMIZERS[optimizer], color=COLOURS[cell],
                  linewidth=1.0)
        axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                          color=COLOURS[cell], alpha=0.15, linewidth=0)
    axis.set_xlabel("Frames (millions)", fontsize=label_size)
    axis.set_ylabel("IQM Reward Rate", fontsize=label_size)
    axis.set_xticks([0, 1, 2, 3, 4, 5])
    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=tick_size)
    for spine in axis.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
    handles = [plt.Line2D([], [], color=COLOURS[c], linewidth=1.0, label=LABELS[c])
               for c in CELLS]
    handles += [plt.Line2D([], [], color="0.3", linewidth=1.0, linestyle=st,
                           label=OPTIMIZER_LABELS[o]) for o, st in OPTIMIZERS.items()]
    figure.legend(handles=handles, loc="outside lower center", frameon=False,
                  fontsize=tick_size, ncol=3, handlelength=1.8, columnspacing=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
