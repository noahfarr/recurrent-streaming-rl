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
OUTPUT = Path("/home/farr/recurrent-streaming-rl/paper/plots/trace_conditioning/return_error.pdf")
CACHE = Path("/home/farr/recurrent-streaming-rl/paper/plots/trace_conditioning/.curves.pkl")
SINCE = "2026-09-06T12:00:00"
ENV_ID = "TraceConditioning"
LAMBDAS = [0.9, 1.0]
CELLS = ["rtu", "min_gru", "ffn"]
LABELS = {"rtu": "RTU", "min_gru": "minGRU", "ffn": "FFN"}
COLOURS = {"rtu": "C3", "min_gru": "C0", "ffn": "C2"}
STYLES = {"RTRL": "-", "TBPTT(1)": "--"}


def mode_of(config):
    target = str((config.get("mode") or {}).get("wrapper", {}).get("_target_", ""))
    return "RTRL" if target.endswith("RTRL") else "TBPTT(1)"


def fetch():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return pickle.loads(CACHE.read_bytes())
    api = wandb.Api(timeout=120)
    runs = api.runs(
        PROJECT,
        filters={"state": "finished", "config.environment.env_id": ENV_ID, "createdAt": {"$gt": SINCE}},
        per_page=500,
    )
    records = defaultdict(list)
    for run in runs:
        config = run.config
        cell = config["groups"]["cell"]
        isi = (config["environment"].get("kwargs") or {}).get("isi")
        lam = config["algorithm"]["trace_lambda"]
        mode = "TBPTT(1)" if cell == "ffn" else mode_of(config)
        history = run.history(keys=["return_error"], samples=100, pandas=False)
        values = [row["return_error"] for row in history if row.get("return_error") is not None]
        if len(values) < 2:
            continue
        records[(isi, lam, cell, mode)].append(float(np.mean(values[-2:])))
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
    isis = sorted({k[0] for k in records})
    label_size = plt.rcParams["axes.labelsize"]
    tick_size = plt.rcParams["xtick.labelsize"]
    figure, axes = plt.subplots(1, len(LAMBDAS), figsize=(FIG_WIDTH, FIG_WIDTH / len(LAMBDAS) * 0.58), constrained_layout=True, sharey=True)
    for axis, lam in zip(np.atleast_1d(axes), LAMBDAS):
        for cell in CELLS:
            for mode, style in STYLES.items():
                if cell == "ffn" and mode == "RTRL":
                    continue
                centres, spreads = [], []
                for isi in isis:
                    values = records.get((isi, lam, cell, mode), [])
                    centre, spread = interquartile_mean(values) if values else (np.nan, np.nan)
                    centres.append(centre)
                    spreads.append(spread)
                centres, spreads = np.array(centres), np.array(spreads)
                label = LABELS[cell] if cell == "ffn" else f"{LABELS[cell]}, {mode}"
                axis.plot(isis, centres, style, color=COLOURS[cell], linewidth=1.0, label=label)
                axis.fill_between(isis, centres - spreads, centres + spreads, color=COLOURS[cell], alpha=0.2, linewidth=0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(isis)
        axis.set_xticklabels([str(i) for i in isis])
        axis.set_title(f"$\\lambda = {lam}$", fontsize=label_size)
        axis.set_xlabel("Interstimulus interval", fontsize=label_size)
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=tick_size)
        for spine in axis.spines.values():
            spine.set_linewidth(plt.rcParams["axes.linewidth"])
    np.atleast_1d(axes)[0].set_ylabel("Return error (MSRE)", fontsize=label_size)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False, fontsize=tick_size)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")
    for key in sorted(records, key=str):
        centre, spread = interquartile_mean(records[key])
        print(key, f"{centre:.4f} ± {spread:.4f} (n={len(records[key])})")


if __name__ == "__main__":
    main()
