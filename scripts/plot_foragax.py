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
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/reward_rate.pdf")
DIPS_OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/dips.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/.curves.pkl")
ENV_ID = "ForagaxNeverEndingRelearning-v1"
SINCE = "2026-09-05T00:00:00"
TOTAL_TIMESTEPS = 5_000_000
PERIOD = 500_000
METRIC = "reward_rate"
PPO_ENTROPY = 0.1
ALGORITHMS = ["stream_q", "intentional_q", "real_time_ppo"]
TITLES = {"stream_q": "stream RQ($\\lambda$)", "intentional_q": "intentional RQ($\\lambda$)", "real_time_ppo": "PPO"}
CELLS = ["rtu", "min_gru", "ffn"]
LABELS = {"rtu": "RTU", "min_gru": "minGRU", "ffn": "FFN"}
COLOURS = dict(zip(CELLS, sns.color_palette("Set1", len(CELLS))))
ALGORITHM_COLOURS = dict(zip(ALGORITHMS, sns.color_palette("Set1", len(ALGORITHMS))))
COMBINED_OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/foragax/reward_rate_combined.pdf")


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.total_timesteps": TOTAL_TIMESTEPS,
            "config.environment.env_id": ENV_ID,
            "createdAt": {"$gt": SINCE},
        },
        order="-created_at",
    )
    records = defaultdict(list)
    for run in runs:
        groups = run.config.get("groups") or {}
        kwargs = (run.config.get("environment") or {}).get("kwargs") or {}
        if kwargs.get("observation_type") != "object":
            continue
        key = (groups.get("algorithm"), groups.get("cell"))
        if key[0] not in ALGORITHMS or key[1] not in CELLS:
            continue
        if key[0] == "real_time_ppo" and (run.config.get("algorithm") or {}).get("entropy_coefficient") != PPO_ENTROPY:
            continue
        if key[0] == "intentional_q":
            target = str((run.config.get("q_optimizer") or {}).get("_target_", ""))
            if not target.endswith("TracePreconditionedIntentional"):
                continue
        steps, values, averages = [], [], []
        for row in run.history(keys=[METRIC, "average_reward"], samples=200, pandas=False):
            if row.get(METRIC) is not None:
                steps.append(row["_step"])
                values.append(row[METRIC])
                averages.append(row.get("average_reward", float("nan")))
        if steps:
            records[key].append((np.array(steps), np.array(values), np.array(averages)))
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    low, high = np.percentile(values, [25, 75], axis=0)
    masked = np.where((values >= low) & (values <= high), values, np.nan)
    return np.nanmean(masked, axis=0)


def draw(axis, curves, colour, label, style="-"):
    if not curves:
        return
    start = max(s.min() for s, *_ in curves)
    limit = min(s.max() for s, *_ in curves)
    grid = np.linspace(start, limit, 100)
    stacked = np.stack([np.interp(grid, s, v) for s, v, *_ in curves])
    centre = interquartile_mean(stacked)
    spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
    axis.plot(grid / 1e6, centre, style, color=colour, linewidth=1.0, label=label)
    axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                      color=colour, alpha=0.15, linewidth=0)


DIP_ALGORITHMS = {"stream_q": "-", "real_time_ppo": "--"}
DIP_LABELS = {"stream_q": "stream RQ($\\lambda$)", "real_time_ppo": "PPO"}


def post_flip_minima(curves):
    flips = np.arange(PERIOD, TOTAL_TIMESTEPS, PERIOD)
    minima = []
    for steps, values, *_ in curves:
        after = np.stack([np.interp(flip + np.array([PERIOD / 5, 2 * PERIOD / 5]), steps, values) for flip in flips])
        minima.append(after.min(axis=1))
    return np.stack(minima)


def draw_dips(axis, records):
    for algorithm, style in DIP_ALGORITHMS.items():
        for cell in CELLS:
            curves = records.get((algorithm, cell), [])
            if not curves:
                continue
            minima = post_flip_minima(curves)
            centre = interquartile_mean(minima)
            spread = minima.std(axis=0) / np.sqrt(minima.shape[0])
            flips = np.arange(1, len(centre) + 1)
            axis.plot(flips, centre, style, color=COLOURS[cell], linewidth=1.0)
            axis.fill_between(flips, centre - spread, centre + spread, color=COLOURS[cell], alpha=0.2, linewidth=0)
    return [plt.Line2D([], [], color=COLOURS[c], linewidth=1.0, label=LABELS[c]) for c in CELLS] + [
        plt.Line2D([], [], color="0.2", linestyle=style, linewidth=1.0, label=DIP_LABELS[a])
        for a, style in DIP_ALGORITHMS.items() if any(records.get((a, c)) for c in CELLS)
    ]


def body_figure(records, label_size, tick_size):
    figure, axes = plt.subplots(
        1, 3, figsize=(FIG_WIDTH, FIG_WIDTH / 3 * 0.90), constrained_layout=True
    )
    figure.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03)

    for axis, index in zip(axes[:2], (1, 2)):
        for boundary in range(PERIOD, TOTAL_TIMESTEPS, PERIOD):
            axis.axvline(boundary / 1e6, color="0.5", linewidth=0.5,
                         linestyle=":", alpha=0.5)
        for algorithm in ALGORITHMS:
            curves = records.get((algorithm, "rtu"), [])
            if not curves:
                continue
            series = [(c[0], c[index]) for c in curves]
            start_step = max(x.min() for x, _ in series)
            limit = min(x.max() for x, _ in series)
            grid = np.linspace(start_step, limit, 150)
            stacked = np.stack([np.interp(grid, x, v) for x, v in series])
            centre = interquartile_mean(stacked)
            spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
            axis.plot(grid / 1e6, centre, color=ALGORITHM_COLOURS[algorithm], linewidth=1.0)
            axis.fill_between(grid / 1e6, centre - spread, centre + spread,
                              color=ALGORITHM_COLOURS[algorithm], alpha=0.2, linewidth=0)
        axis.set_xlabel("Frames (millions)", fontsize=label_size)
        axis.set_xticks([0, 1, 2, 3, 4, 5])

    axes[0].set_ylabel("IQM reward rate", fontsize=label_size)
    axes[1].set_ylabel("Average reward", fontsize=label_size)

    for cell in CELLS:
        curves = records.get(("stream_q", cell), [])
        if not curves:
            continue
        minima = post_flip_minima(curves)
        centre = interquartile_mean(minima)
        spread = minima.std(axis=0) / np.sqrt(minima.shape[0])
        flips = np.arange(1, len(centre) + 1)
        axes[2].plot(flips, centre, color=COLOURS[cell], linewidth=1.0)
        axes[2].fill_between(flips, centre - spread, centre + spread,
                             color=COLOURS[cell], alpha=0.2, linewidth=0)
    axes[2].set_xlabel("Flip", fontsize=label_size)
    axes[2].set_ylabel("Post-flip minimum", fontsize=label_size)
    axes[2].set_xticks(range(1, TOTAL_TIMESTEPS // PERIOD))

    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=tick_size)
        for spine in axis.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])
        axis.set_box_aspect(0.72)

    algorithm_handles = [plt.Line2D([], [], color=ALGORITHM_COLOURS[a], linewidth=1.0,
                                    label=TITLES[a]) for a in ALGORITHMS]
    cell_handles = [plt.Line2D([], [], color=COLOURS[c], linewidth=1.0, label=LABELS[c])
                    for c in CELLS]
    axes[1].legend(handles=algorithm_handles, loc="lower right", frameon=False,
                   fontsize=tick_size, handlelength=1.4, labelspacing=0.25)
    axes[2].legend(handles=cell_handles, loc="lower right", frameon=False,
                   fontsize=tick_size, handlelength=1.4, labelspacing=0.25)
    out = DIPS_OUTPUT.with_name("foragax_body.pdf")
    figure.savefig(out)
    plt.close(figure)
    print(f"wrote {out}")


def main():
    records = fetch()
    for key in sorted(records):
        print(f"  {key}: {len(records[key])} seeds")
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]

    single, axis = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * 0.62, FIG_WIDTH * 0.62 / GOLDEN_RATIO),
        constrained_layout=True,
    )
    for boundary in range(PERIOD, TOTAL_TIMESTEPS, PERIOD):
        axis.axvline(boundary / 1e6, color="0.5", linewidth=0.5, linestyle=":", alpha=0.5)
    for algorithm in ALGORITHMS:
        draw(axis, records.get((algorithm, "rtu"), []), ALGORITHM_COLOURS[algorithm],
             TITLES[algorithm])
    axis.set_xlabel("Frames (millions)", fontsize=label_size)
    axis.set_ylabel("IQM Reward Rate", fontsize=label_size)
    axis.set_xticks([0, 1, 2, 3, 4, 5])
    axis.set_ylim(-0.6, 2.1)
    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=tick_size)
    for spine in axis.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
    handles = [plt.Line2D([], [], color=ALGORITHM_COLOURS[a], linewidth=1.0,
                          label=TITLES[a]) for a in ALGORITHMS]
    single.legend(handles=handles, loc="outside upper center", frameon=False,
                  fontsize=tick_size, ncol=len(ALGORITHMS), handlelength=1.8)
    COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    single.savefig(COMBINED_OUTPUT)
    plt.close(single)
    print(f"wrote {COMBINED_OUTPUT}")

    columns = 2
    figure, grid_axes = plt.subplots(
        2, columns,
        figsize=(FIG_WIDTH, FIG_WIDTH / columns * 0.75 * 2.1),
        constrained_layout=True,
    )
    axes = grid_axes.ravel()
    for axis, algorithm in zip(axes[:-1], ALGORITHMS):
        for boundary in range(PERIOD, TOTAL_TIMESTEPS, PERIOD):
            axis.axvline(boundary / 1e6, color="0.5", linewidth=0.5, linestyle=":", alpha=0.5)
        for cell in CELLS:
            draw(axis, records.get((algorithm, cell), []), COLOURS[cell], LABELS[cell])
        title = TITLES[algorithm]
        if algorithm == "real_time_ppo":
            title = f"{title} (\\gls*{{rtu}} only)" if False else f"{title} (RTU only)"
        axis.set_title(title, fontsize=label_size)
        axis.set_xlabel("Frames (millions)", fontsize=label_size)
        axis.set_xticks([0, 1, 2, 3, 4, 5])
        axis.set_ylim(-0.6, 2.1)
    axes[0].set_ylabel("IQM Reward Rate", fontsize=label_size)
    axes[2].set_ylabel("IQM Reward Rate", fontsize=label_size)
    axes[1].set_yticklabels([])
    legend_axis = axes[-1]
    legend_axis.axis("off")
    for axis in axes[:-1]:
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=tick_size)
        for spine in axis.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])
        axis.set_box_aspect(0.75)
    handles = [plt.Line2D([], [], color=COLOURS[c], linewidth=1.0, label=LABELS[c]) for c in CELLS]
    legend_axis.legend(handles=handles, loc="center", frameon=False,
                       fontsize=tick_size, handlelength=1.8, labelspacing=0.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")
    body_figure(records, label_size, tick_size)

    width = FIG_WIDTH * 0.36
    small, axis = plt.subplots(1, 1, figsize=(width, width * 1.02), constrained_layout=True)
    dip_handles = draw_dips(axis, records)
    small.legend(handles=dip_handles, loc="outside lower center", frameon=False,
                 fontsize=tick_size, ncol=2, handlelength=1.6, columnspacing=1.0,
                 labelspacing=0.3)
    axis.set_xlabel("Flip", fontsize=label_size)
    axis.set_ylabel("Post-flip minimum", fontsize=label_size)
    axis.set_xticks(range(1, TOTAL_TIMESTEPS // PERIOD))
    axis.set_ylim(-0.6, 2.7)
    axis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=tick_size)
    for spine in axis.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
    axis.set_box_aspect(0.75)
    small.savefig(DIPS_OUTPUT)
    print(f"wrote {DIPS_OUTPUT}")


if __name__ == "__main__":
    main()
