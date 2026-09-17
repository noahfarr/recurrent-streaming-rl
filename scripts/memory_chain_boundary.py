import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from styles import FIG_WIDTH, GOLDEN_RATIO, apply as apply_style

apply_style()

ROOT = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain")
OUTPUT = ROOT / "horizon_boundary.pdf"
ALGORITHMS = [
    ("qrc", ROOT / ".grid.pkl", "QRC($\\lambda$)"),
    ("stream_q", ROOT / ".grid_stream_q.pkl", "stream $Q(\\lambda)$"),
    ("intentional_q", ROOT / ".grid_intentional_q.pkl", "intentional $Q(\\lambda)$"),
]
LENGTHS = [16, 32, 64, 96, 128, 192, 256]
TICKS = [16, 32, 64, 128, 256]
LAMBDAS = [0.8, 0.9, 0.98, 0.995, 1.0]
MODES = ["TBPTT(1)", "RTRL"]
GAMMA = 0.999
THRESHOLD = 0.5


def boundary(records, trace_lambda, mode):
    deepest = np.nan
    for length in LENGTHS:
        scores = records.get((length, trace_lambda, mode))
        if not scores:
            continue
        rate = float(np.mean(np.array(scores) > THRESHOLD))
        if rate >= 0.5:
            deepest = length
    return deepest


def main():
    palette = sns.color_palette("Set1", 3)
    styles = {"TBPTT(1)": (palette[1], "-"), "RTRL": (palette[0], (0, (1.4, 1.2)))}
    positions = np.arange(len(LAMBDAS))
    horizon = [1.0 / (1.0 - GAMMA * value) for value in LAMBDAS]

    figure, axes = plt.subplots(
        1, len(ALGORITHMS), sharey=True,
        figsize=(FIG_WIDTH, FIG_WIDTH / len(ALGORITHMS) / GOLDEN_RATIO * 1.65),
        constrained_layout=True,
    )

    for axis, (name, path, label) in zip(axes, ALGORITHMS):
        axis.plot(positions, horizon, color="0.62", linewidth=0.8,
                  linestyle=(0, (4.0, 2.0)), zorder=1)
        if path.exists():
            records = pickle.loads(path.read_bytes())
            for mode in MODES:
                colour, style = styles[mode]
                values = [boundary(records, value, mode) for value in LAMBDAS]
                axis.plot(positions, values, color=colour, linewidth=1.3,
                          linestyle=style, zorder=3)
        axis.set_yscale("log", base=2)
        axis.set_yticks(TICKS)
        axis.set_yticklabels([str(value) for value in TICKS])
        axis.set_yticks(LENGTHS, minor=True)
        axis.set_ylim(12, 400)
        axis.set_xticks(positions)
        axis.set_xticklabels([str(value) for value in LAMBDAS], fontsize=6)
        axis.set_title(label, fontsize=7.5)
        axis.tick_params(labelsize=6.5)
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)

    axes[0].set_ylabel(r"Chain length $L$", fontsize=8)

    handles = [
        plt.Line2D([], [], color=styles["TBPTT(1)"][0], linewidth=1.3,
                   linestyle=styles["TBPTT(1)"][1], label="TBPTT(1)"),
        plt.Line2D([], [], color=styles["RTRL"][0], linewidth=1.3,
                   linestyle=styles["RTRL"][1], label="RTRL"),
        plt.Line2D([], [], color="0.62", linewidth=0.8, linestyle=(0, (4.0, 2.0)),
                   label=r"$1/(1-\gamma\lambda)$"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
                  fontsize=7, handlelength=2.2, columnspacing=2.0)
    figure.supxlabel(r"Trace decay $\lambda$", fontsize=8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")

    for name, path, label in ALGORITHMS:
        if not path.exists():
            continue
        records = pickle.loads(path.read_bytes())
        for mode in MODES:
            values = [boundary(records, value, mode) for value in LAMBDAS]
            print(f"  {name:<15} {mode:<9} " + " ".join(f"{v:>6.0f}" for v in values))


if __name__ == "__main__":
    main()
