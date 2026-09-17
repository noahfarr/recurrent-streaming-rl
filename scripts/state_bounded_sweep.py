import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from styles import FIG_WIDTH, apply as apply_style

apply_style()

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/state_bounded.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/.state_bounded.pkl")
SINCE = "2026-09-05T19:50:00"
ALGORITHMS = {"stream_q": "ObGD", "block_q": "block ObGD", "state_q": "state-bounded"}
COLOURS = {"stream_q": "0.45", "block_q": "0.75", "state_q": "C3"}
SWEEP_ALGORITHMS = ["stream_q", "state_q"]
SWEEP_LENGTHS = [32, 64]
STYLES = {32: "-", 64: "--"}
BAR_LENGTHS = [64, 128, 192]


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api(timeout=120)
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.environment.env_id": "MemoryChain-bsuite",
            "config.total_timesteps": 500000,
            "createdAt": {"$gt": SINCE},
            "displayName": {"$regex": "^(state_q|stream_q)"},
        },
        per_page=500,
    )
    records = defaultdict(list)
    for run in runs:
        config = run.config
        groups = config["groups"]
        score = run.summary.get("score")
        if score is None:
            continue
        length = config["environment"]["env_params"]["max_steps_in_episode"] - 1
        optimizer = config["q_optimizer"]["cfg"]
        if optimizer.get("relative") and not optimizer.get("block"):
            continue
        algorithm = "block_q" if optimizer.get("block") else groups["algorithm"]
        records[(algorithm, groups["cell"], length, optimizer["lr"], optimizer.get("kappa_state", 1.0))].append(score)
    records = dict(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(records))
    return records


def interquartile_mean(values):
    values = np.asarray(values)
    low, high = np.percentile(values, [25, 75])
    kept = values[(values >= low) & (values <= high)]
    return kept.mean(), kept.std() / np.sqrt(len(kept))


def main():
    records = fetch()
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH / 2 * 0.5), constrained_layout=True)
    lrs = sorted({k[3] for k in records if k[1] == "rtu" and k[4] == 1.0})
    for algorithm in SWEEP_ALGORITHMS:
        colour = COLOURS[algorithm]
        for length in SWEEP_LENGTHS:
            centres, spreads = [], []
            for lr in lrs:
                values = records.get((algorithm, "rtu", length, lr, 1.0), [])
                centre, spread = interquartile_mean(values) if values else (np.nan, np.nan)
                centres.append(centre)
                spreads.append(spread)
            centres, spreads = np.array(centres), np.array(spreads)
            left.plot(lrs, centres, STYLES[length], color=colour, linewidth=1.0,
                      label=f"{ALGORITHMS[algorithm]}, $L={length}$")
            left.fill_between(lrs, centres - spreads, centres + spreads, color=colour, alpha=0.2, linewidth=0)
    left.set_xscale("log")
    left.set_xlabel("Step size $\\alpha$", fontsize=label_size)
    left.set_ylabel("IQM score", fontsize=label_size)
    left.set_title("RTU", fontsize=label_size)
    left.legend(frameon=False, fontsize=tick_size, loc="center left")
    width = 0.27
    for offset, (algorithm, colour) in zip((-width, 0.0, width), COLOURS.items()):
        centres, spreads = [], []
        for length in BAR_LENGTHS:
            values = records.get((algorithm, "min_gru", length, 1.0, 1.0), [])
            centre, spread = interquartile_mean(values) if values else (np.nan, np.nan)
            centres.append(centre)
            spreads.append(spread)
        right.bar(np.arange(len(BAR_LENGTHS)) + offset, centres, width, yerr=spreads, color=colour,
                  label=ALGORITHMS[algorithm], error_kw={"linewidth": 0.6})
    right.set_xticks(np.arange(len(BAR_LENGTHS)))
    right.set_xticklabels([f"$L={l}$" for l in BAR_LENGTHS])
    right.set_title("minGRU, $\\alpha = 1$", fontsize=label_size)
    right.legend(frameon=False, fontsize=tick_size, loc="upper center", ncol=3)
    for axis in (left, right):
        axis.set_ylim(-0.1, 1.05)
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=tick_size)
        for spine in axis.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])
    right.set_ylim(-0.1, 1.45)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")
    for key in sorted(records, key=str):
        centre, spread = interquartile_mean(records[key])
        print(key, f"{centre:.2f} ± {spread:.2f} (n={len(records[key])})")


if __name__ == "__main__":
    main()
