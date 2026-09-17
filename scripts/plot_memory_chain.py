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
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/.cache.pkl")
LENGTHS = [16, 32, 64, 96, 128, 192, 256]
LAMBDAS = [0.9, 0.98, 0.995, 1.0]
MODES = ["TBPTT(1)", "RTRL"]
COLOURS = dict(zip(MODES, sns.color_palette("Set1", 2)))
THRESHOLD = 0.5
REFERENCE_LENGTH = 128


def mode_of(run):
    target = str(run.config.get("mode", {}).get("wrapper", {}).get("_target_", ""))
    return "RTRL" if target.endswith("RTRL") else "TBPTT(1)"


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.total_timesteps": 500000,
            "config.num_seeds": {"$in": [16, 32]},
            "config.environment.env_id": "MemoryChain-bsuite",
        },
        order="-created_at",
    )
    records = []
    for run in list(runs)[:1200]:
        config = run.config
        environment = config.get("environment", {})
        if environment.get("wrappers"):
            continue
        if "MinGRU" not in str(config.get("cell", {}).get("config", {}).get("_target_", "")):
            continue
        algorithm = config.get("algorithm", {})
        if algorithm.get("gamma") != 0.999:
            continue
        history = run.history(keys=["training/episode_returns"], samples=120)
        if history.empty:
            continue
        records.append(
            dict(
                mode=mode_of(run),
                length=((environment.get("env_params") or {}).get("max_steps_in_episode") or 1) - 1,
                trace_lambda=algorithm.get("trace_lambda"),
                steps=history["_step"].to_numpy(),
                returns=history["training/episode_returns"].to_numpy(),
                final=run.summary.get("score"),
            )
        )
    CACHE.write_bytes(pickle.dumps(records))
    return records


def solve_rate(values):
    values = [v for v in values if v is not None]
    return (np.mean([v > THRESHOLD for v in values]) if values else np.nan, len(values))


def wilson(rate, count, z=1.0):
    if count == 0:
        return 0.0, 0.0
    centre = (rate + z * z / (2 * count)) / (1 + z * z / count)
    spread = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def panel_length(axis, records):
    for mode in MODES:
        grouped = defaultdict(list)
        for record in records:
            if record["mode"] == mode and record["trace_lambda"] == 1.0:
                grouped[record["length"]].append(record["final"])
        lengths = sorted(grouped)
        if not lengths:
            continue
        rates, lows, highs = [], [], []
        for length in lengths:
            rate, count = solve_rate(grouped[length])
            low, high = wilson(rate, count)
            rates.append(rate)
            lows.append(low)
            highs.append(high)
        axis.plot(lengths, rates, marker="o", markersize=2.5, color=COLOURS[mode], label=mode)
        axis.fill_between(lengths, lows, highs, color=COLOURS[mode], alpha=0.18, linewidth=0)
    axis.set_xscale("log", base=2)
    ticks = [16, 32, 64, 128, 256]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(v) for v in ticks])
    axis.minorticks_off()
    axis.set_xlabel("chain length $L$")
    axis.set_ylabel("seeds solved")
    axis.set_ylim(-0.03, 1.03)


def panel_lambda(axis, records):
    for mode in MODES:
        grouped = defaultdict(list)
        for record in records:
            if record["mode"] == mode and record["length"] == REFERENCE_LENGTH:
                grouped[record["trace_lambda"]].append(record["final"])
        lambdas = sorted(x for x in grouped if x is not None)
        if not lambdas:
            continue
        rates, lows, highs = [], [], []
        for value in lambdas:
            rate, count = solve_rate(grouped[value])
            low, high = wilson(rate, count)
            rates.append(rate)
            lows.append(low)
            highs.append(high)
        gaps = [max(1.0 - value, 1e-3) for value in lambdas]
        axis.plot(gaps, rates, marker="o", markersize=2.5, color=COLOURS[mode], label=mode)
        axis.fill_between(gaps, lows, highs, color=COLOURS[mode], alpha=0.18, linewidth=0)
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xticks([1e-1, 2e-2, 5e-3, 1e-3])
    axis.set_xticklabels(["0.9", "0.98", "0.995", "1"])
    axis.minorticks_off()
    axis.set_xlabel("trace decay $\\lambda$")
    axis.set_ylabel("seeds solved")
    axis.set_ylim(-0.03, 1.03)


def panel_curves(axis, records):
    for mode in MODES:
        curves = [
            (r["steps"], r["returns"])
            for r in records
            if r["mode"] == mode and r["length"] == REFERENCE_LENGTH and r["trace_lambda"] == 1.0
        ]
        if not curves:
            continue
        limit = min(s.max() for s, _ in curves)
        grid = np.linspace(0, limit, 60)
        values = np.stack([np.interp(grid, s, v) for s, v in curves])
        low, high = np.percentile(values, [25, 75], axis=0)
        axis.fill_between(grid / 1e3, low, high, color=COLOURS[mode], alpha=0.18, linewidth=0)
        axis.plot(grid / 1e3, np.median(values, axis=0), color=COLOURS[mode], label=mode)
    axis.set_xlabel("steps ($\\times 10^3$)")
    axis.set_ylabel("episodic return")
    axis.set_ylim(-0.35, 1.05)


def main():
    records = fetch()
    print(f"fetched {len(records)} runs")

    width = FIG_WIDTH
    height = (width / 3) / GOLDEN_RATIO
    figure, axes = plt.subplots(1, 3, figsize=(width, height))

    panel_length(axes[0], records)
    panel_lambda(axes[1], records)
    panel_curves(axes[2], records)

    for axis, label in zip(axes, "abc"):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(f"({label})", loc="left", fontsize=8)
    axes[0].legend(frameon=False, handlelength=1.2, borderpad=0.2,
                   labelspacing=0.25, loc="lower left")

    figure.tight_layout(pad=0.4, w_pad=1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "memory_chain_lambda.pdf"
    figure.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
