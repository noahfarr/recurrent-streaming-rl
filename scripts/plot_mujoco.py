import pickle
import subprocess
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
from styles import FIG_WIDTH, GOLDEN_RATIO, SUBPLOT_HEIGHT, apply as apply_style

apply_style()

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/mujoco/learning_curves.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/mujoco/.curves.pkl")
DATA = Path("/home/farr/recurrent-streaming-rl/paper/data/mujoco")
REMOTES = ["ias:~/slurmpilot/jobs/main/2026-09-05_19-37-41/outputs/", "ias:~/slurmpilot/jobs/main/2026-09-06_22-23-52/outputs/", "ias:~/slurmpilot/jobs/main/2026-09-07_08-21-10/outputs/", "ias:~/slurmpilot/jobs/main/2026-09-07_10-54-59/outputs/"]
NAMESPACE = "gymnasium"
SINCE = "2026-09-05T12:00:00"
STREAMING_SINCE = "2026-09-07T12:00:00"
TRACE_LAMBDA = 0.99
INTENTIONAL_ETA = 0.01
PPO_BODY = "streaming"
INTENTIONAL_SINCE = "2026-09-11T18:00:00"
PPO_SINCE = "2026-09-11T08:00:00"
TOTAL_TIMESTEPS = 5_000_000
ALGORITHMS = ["stream_ac", "intentional_ac", "real_time_ppo"]
LABELS = {"stream_ac": "stream RAC($\\lambda$)",
          "intentional_ac": "intentional RAC($\\lambda$)", "real_time_ppo": "PPO"}
COLOURS = dict(zip(ALGORITHMS, sns.color_palette("Set1", len(ALGORITHMS))))
TASKS = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
MODES = ["P", "V"]
CELLS = {"rtu": "RTU", "ffn": "FFN", "min_gru": "MinGRU"}
MEMORYLESS_ALGORITHMS = ["stream_ac", "intentional_ac"]


def algorithm_of(name):
    for a in ("intentional_ac", "stream_ac", "real_time_ppo"):
        if name.startswith(a):
            return a
    return None


def overrides_of(run_dir):
    values = {}
    for line in (run_dir / ".hydra" / "overrides.yaml").read_text().splitlines():
        line = line.strip().lstrip("- ").lstrip("+")
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def fetch_files():
    if "--sync" in sys.argv:
        DATA.mkdir(parents=True, exist_ok=True)
        for index, remote in enumerate(REMOTES):
            target = DATA / f"batch{index}"
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--include=*/", "--include=metrics.npz", "--include=overrides.yaml",
                            "--exclude=*", remote, str(target)], check=False)
    records = defaultdict(list)
    newest = {}
    for metrics in DATA.rglob("metrics.npz"):
        overrides = overrides_of(metrics.parent)
        if overrides.get("logger") != "file":
            continue
        identity = (overrides["algorithm"], overrides["environment"], overrides["environment.kwargs.mode"], overrides.get("cell", "rtu"), overrides.get("seed", "0"))
        if identity not in newest or metrics.stat().st_mtime > newest[identity].stat().st_mtime:
            newest[identity] = metrics
    for metrics in newest.values():
        overrides = overrides_of(metrics.parent)
        algorithm = overrides["algorithm"]
        task = {"mujoco/ant": "Ant", "mujoco/halfcheetah": "HalfCheetah",
                "mujoco/hopper": "Hopper", "mujoco/walker2d": "Walker2d"}[overrides["environment"]]
        mode = overrides["environment.kwargs.mode"]
        data = np.load(metrics)
        steps = data["steps"]
        for key in data.files:
            if key.startswith("training/episode_returns/"):
                values = data[key].reshape(len(steps), -1)[:, 0]
                keep = np.isfinite(values)
                if keep.sum() > 1:
                    cell = CELLS.get(overrides.get("cell", "rtu"), "RTU")
                    records[(task, mode, algorithm, cell)].append((steps[keep], values[keep]))
    return records


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.total_timesteps": TOTAL_TIMESTEPS,
            "config.environment.namespace": NAMESPACE,
            "createdAt": {"$gt": SINCE},
        },
        order="-created_at",
    )
    records = defaultdict(list)
    for run in runs:
        algorithm = algorithm_of(run.name)
        if algorithm is None:
            continue
        config = run.config
        algorithm_config = config.get("algorithm") or {}
        if algorithm == "real_time_ppo":
            if config.get("body") != PPO_BODY:
                continue
            if run.created_at < PPO_SINCE:
                continue
        else:
            if algorithm_config.get("trace_lambda") != TRACE_LAMBDA:
                continue
            if algorithm == "intentional_ac":
                target = str((config.get("actor_optimizer") or {}).get("_target_", ""))
                if not target.endswith("TracePreconditionedIntentional"):
                    continue
                eta = ((config.get("actor_optimizer") or {}).get("cfg") or {}).get("eta")
                if eta != INTENTIONAL_ETA:
                    continue
            if algorithm_config.get("entropy_coefficient") == 0.0:
                continue
            if config.get("width") is not None or config.get("blocks") is not None:
                continue
            if str((config.get("actor_optimizer") or {}).get("_target_", "")).endswith(
                ("BlockwiseObGD", "NormalizedObGD")
            ):
                continue
        environment = config.get("environment") or {}
        task = str(environment.get("env_id", "")).split("-")[0]
        mode = (environment.get("kwargs") or {}).get("mode")
        cell = CELLS.get(str((config.get("groups") or {}).get("cell")), "RTU")
        history = run.history(keys=["training/episode_returns"], samples=60, pandas=False)
        steps, values = [], []
        for row in history:
            value = row.get("training/episode_returns")
            if value is not None:
                steps.append(row["_step"])
                values.append(value)
        if steps:
            records[(task, mode, algorithm, cell)].append((np.array(steps), np.array(values)))
    if "--files" in sys.argv:
        for key, curves in fetch_files().items():
            records[key].extend(curves)
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    low, high = np.nanpercentile(values, [25, 75], axis=0)
    masked = np.where((values >= low) & (values <= high), values, np.nan)
    return np.nanmean(masked, axis=0)


def draw(axis, curves, colour, minimum_seeds=10, linestyle="solid", band=True):
    if not curves:
        return
    start = max(s.min() for s, _ in curves)
    limit = max(s.max() for s, _ in curves)
    grid = np.linspace(start, limit, 50 if band else 22)
    stacked = np.stack([
        np.where(grid <= s.max(), np.interp(grid, s, v), np.nan) for s, v in curves
    ])
    covered = np.sum(np.isfinite(stacked), axis=0) >= minimum_seeds
    grid, stacked = grid[covered], stacked[:, covered]
    centre = interquartile_mean(stacked)
    spread = np.nanstd(stacked, axis=0) / np.sqrt(np.sum(np.isfinite(stacked), axis=0))
    width = 1.0 if band else 0.7
    axis.plot(grid / 1e6, centre, color=colour, linewidth=width, linestyle=linestyle,
              alpha=1.0 if band else 0.5, zorder=2 if band else 1)
    if band:
        axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                          color=colour, alpha=0.2, linewidth=0)


def main():
    records = fetch()
    print(f"cells fetched: {len(records)}")
    for key in sorted(records):
        print(f"  {key}: {len(records[key])} seeds")
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, axes = plt.subplots(
        len(MODES), len(TASKS),
        figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * 1.42 * len(MODES)),
        constrained_layout=True,
    )
    for row, mode in enumerate(MODES):
        for column, task in enumerate(TASKS):
            axis = axes[row, column]
            for algorithm in ALGORITHMS:
                draw(axis, records.get((task, mode, algorithm, "RTU"), []), COLOURS[algorithm])
            for algorithm in MEMORYLESS_ALGORITHMS:
                draw(axis, records.get((task, mode, algorithm, "FFN"), []),
                     COLOURS[algorithm], linestyle=(0, (2.2, 1.6)), band=False)
            if row == 0:
                axis.set_title(task, fontsize=label_size)
            if column == 0:
                axis.set_ylabel(mode, fontsize=label_size, rotation=0, labelpad=10)
            axis.grid(True, alpha=0.2)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=tick_size)
            axis.set_xticks([0, 2.5, 5])
            for spine in axis.spines.values():
                spine.set_linewidth(plt.rcParams["axes.linewidth"])
            axis.set_box_aspect(0.66)
    handles = [plt.Line2D([], [], color=COLOURS[a], linewidth=1.0, label=LABELS[a])
               for a in ALGORITHMS]
    handles += [plt.Line2D([], [], color="0.35", linewidth=1.0, label="RTU"),
                plt.Line2D([], [], color="0.35", linewidth=0.7, alpha=0.6,
                           linestyle=(0, (2.2, 1.6)), label="FFN")]
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles),
                  frameon=False, fontsize=tick_size)
    figure.supxlabel("Number of Frames (in millions)", fontsize=label_size)
    figure.supylabel("IQM Episode Return", fontsize=label_size)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
