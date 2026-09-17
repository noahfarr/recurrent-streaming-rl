import json
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
from styles import FIG_WIDTH, GOLDEN_RATIO, SUBPLOT_HEIGHT, apply as apply_style

apply_style()

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/popgym/learning_curves.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/popgym/.curves.pkl")
INITIAL = Path("/home/farr/recurrent-streaming-rl/paper/plots/popgym/initial_returns.json")
SINCE = "2026-09-03T18:00:00"
ALGORITHMS = ["stream_q", "qrc", "intentional_q", "real_time_ppo"]
CEILING_ALGORITHMS = ["stream_q", "qrc", "intentional_q", "ppo"]
INITIAL_KEY = {"real_time_ppo": "ppo"}
TRACE_LAMBDA = {"stream_q": 0.8, "qrc": 0.95, "intentional_q": 0.8}
OPTIMIZER = {"stream_q": "ObGD", "qrc": "sgd", "intentional_q": "Intentional"}
SEEDS_PER_ARM = 10
LABELS = {"qrc": "RQRC($\\lambda$)", "stream_q": "stream RQ($\\lambda$)",
          "intentional_q": "intentional RQ($\\lambda$)", "real_time_ppo": "PPO"}
COLOURS = dict(zip(ALGORITHMS, sns.color_palette("Set1", len(ALGORITHMS))))
TASKS = ["RepeatPrevious", "StatelessCartPole", "CountRecall", "Minesweeper",
         "Battleship", "NoisyStatelessCartPole", "RepeatFirst", "Autoencode",
         "HigherLower", "MultiArmedBandit", "Concentration"]
MINIMUM = {"StatelessCartPole": 0.0, "NoisyStatelessCartPole": 0.0,
           "Battleship": -64 / 52, "Minesweeper": -13 / 24 - 0.5714285714285714}
DEFAULT_MINIMUM = -1.0

MAXIMUM = {"RepeatPrevious": 1.0, "CountRecall": 1.0, "RepeatFirst": 1.0,
           "Autoencode": 1.0, "StatelessCartPole": 1.0, "NoisyStatelessCartPole": 1.0,
           "Minesweeper": 1.0, "Battleship": 1.0, "HigherLower": 0.507,
           "Concentration": 0.816, "MultiArmedBandit": 0.824}


def normalise(task, value):
    low = MINIMUM.get(task, DEFAULT_MINIMUM)
    return (value - low) / (MAXIMUM[task] - low)


CEILING = {}


def algorithm_of(name):
    for a in ("intentional_q", "stream_q", "qrc", "real_time_ppo", "ppo"):
        if name.startswith(a):
            return a
    return None


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.num_seeds": 10,
            "config.num_epochs": 50,
            "config.total_timesteps": 10000000,
            "config.environment.namespace": "popgymnax",
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
        cell = "FFN"
        if config.get("cell"):
            cell = str(config["cell"].get("config", {}).get("_target_", "")).split(".")[-1]
            cell = cell.replace("Config", "")
        if run.created_at < SINCE:
            continue
        expected = TRACE_LAMBDA.get(algorithm)
        if expected is not None:
            if (config.get("algorithm") or {}).get("trace_lambda") != expected:
                continue
        wanted = OPTIMIZER.get(algorithm)
        if wanted is not None:
            source = "q_optimizer"
            target = str((config.get(source) or {}).get("_target_", ""))
            if target.split(".")[-1] != wanted:
                continue
        task = (config.get("environment") or {}).get("env_id")
        if len(records[(task, algorithm, cell)]) >= SEEDS_PER_ARM:
            continue
        history = run.history(keys=["training/episode_returns"], samples=60, pandas=False)
        steps, values = [], []
        for row in history:
            value = row.get("training/episode_returns")
            if value is not None:
                steps.append(row["_step"])
                values.append(value)
        if steps:
            records[(task, algorithm, cell)].append((np.array(steps), np.array(values)))
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    low, high = np.percentile(values, [25, 75], axis=0)
    masked = np.where((values >= low) & (values <= high), values, np.nan)
    return np.nanmean(masked, axis=0)


def draw(axis, curves, colour, task=None, initial=None):
    if not curves:
        return
    if initial:
        if len(initial) != len(curves):
            initial = [float(np.mean(initial))] * len(curves)
        curves = [(np.concatenate([[0.0], s]), np.concatenate([[i], v]))
                  for (s, v), i in zip(curves, initial)]
    start = max(s.min() for s, _ in curves)
    limit = min(s.max() for s, _ in curves)
    grid = np.linspace(start, limit, 50)
    stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
    if task is not None:
        stacked = normalise(task, stacked)
    centre = interquartile_mean(stacked)
    spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
    axis.plot(grid / 1e6, centre, color=colour, linewidth=1.0)
    axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                      color=colour, alpha=0.2, linewidth=0)


def empirical_ceiling(records, task):
    finals = []
    for algorithm in CEILING_ALGORITHMS:
        curves = records.get((task, algorithm, "FFN"), [])
        if not curves:
            continue
        start = max(s.min() for s, _ in curves)
        limit = min(s.max() for s, _ in curves)
        grid = np.linspace(start, limit, 50)
        stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
        finals.append(interquartile_mean(stacked)[-1])
    return max(finals) if finals else None


CELL_ARGS = {"--cell=RTU": "RTU", "--cell=MinGRU": "MinGRU", "--cell=FFN": "FFN"}


def main():
    cell = next((CELL_ARGS[a] for a in sys.argv[1:] if a in CELL_ARGS), "RTU")
    output = OUTPUT if cell == "RTU" else OUTPUT.with_name(f"learning_curves_{cell.lower()}.pdf")
    records = fetch()
    initial = json.loads(INITIAL.read_text()) if (INITIAL.exists() and cell == "RTU") else {}
    print(f"cells fetched: {len(records)}, initial points: {len(initial)}")
    for task in TASKS:
        if task not in CEILING:
            value = empirical_ceiling(records, task)
            if value is not None:
                CEILING[task] = value
                print(f"  empirical ceiling {task}: {value:+.3f}")
    columns = 4
    rows = int(np.ceil(len(TASKS) / columns))
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * 1.15 * rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.04)
    flat = axes.ravel()
    for index, task in enumerate(TASKS):
        axis = flat[index]
        for algorithm in ALGORITHMS:
            entry = initial.get(f"{task}/{INITIAL_KEY.get(algorithm, algorithm)}", {})
            draw(axis, records.get((task, algorithm, cell), []), COLOURS[algorithm],
                 task, entry.get("behaviour_seeds"))
        for algorithm in ALGORITHMS:
            if algorithm == "real_time_ppo":
                continue
            curves = records.get((task, algorithm, "FFN"), [])
            if not curves:
                continue
            start = max(c.min() for c, _ in curves)
            limit = min(c.max() for c, _ in curves)
            grid = np.linspace(start, limit, 22)
            stacked = normalise(task, np.stack([np.interp(grid, c, v) for c, v in curves]))
            axis.plot(grid / 1e6, interquartile_mean(stacked), color=COLOURS[algorithm],
                      linewidth=0.7, linestyle=(0, (2.2, 1.6)), alpha=0.5, zorder=1)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(task, fontsize=label_size)
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=tick_size,
                         labelbottom=index + columns >= len(TASKS),
                         labelleft=index % columns == 0)
        axis.set_xticks([0, 5, 10])
        axis.set_yticks([0.0, 0.5, 1.0])
        for spine in axis.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])
        axis.set_box_aspect(1 / GOLDEN_RATIO)
    legend_axis = flat[len(TASKS)]
    legend_axis.axis("off")
    for axis in flat[len(TASKS) + 1:]:
        axis.set_visible(False)
    present = [a for a in ALGORITHMS
               if any(records.get((task, a, cell)) for task in TASKS)]
    handles = [plt.Line2D([], [], color=COLOURS[a], linewidth=1.0, label=LABELS[a])
               for a in present]
    handles += [plt.Line2D([], [], color="0.3", linewidth=1.0, label="RTU"),
                plt.Line2D([], [], color="0.3", linewidth=0.8,
                           linestyle=(0, (2.2, 1.6)), alpha=0.6, label="FFN")]
    legend_axis.legend(handles=handles, loc="center", frameon=False,
                       fontsize=tick_size, handlelength=1.8, borderpad=0.2,
                       labelspacing=0.6)
    figure.supxlabel("Number of Frames (in millions)", fontsize=label_size)
    figure.supylabel("Normalised IQM Episode Return", fontsize=label_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
