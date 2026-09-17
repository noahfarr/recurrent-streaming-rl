import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from styles import FIG_WIDTH, GOLDEN_RATIO, apply as apply_style

apply_style()

ROOT = Path("/home/farr/recurrent-streaming-rl/paper/plots/memory_chain")
OUTPUT = ROOT / "horizon_difference.pdf"
ALGORITHMS = [
    ("stream_q", ROOT / ".grid_stream_q.pkl", "stream $RQ(\\lambda)$"),
    ("qrc", ROOT / ".grid.pkl", "RQRC($\\lambda$)"),
    ("intentional_q", ROOT / ".grid_intentional_q.pkl", "intentional $RQ(\\lambda)$"),
]
LENGTHS = [16, 32, 64, 96, 128, 192, 256]
LAMBDAS = [0.8, 0.9, 0.98, 0.995, 1.0]
THRESHOLD = 0.5


def matrix(records, mode):
    grid = np.full((len(LENGTHS), len(LAMBDAS)), np.nan)
    for row, length in enumerate(LENGTHS):
        for column, value in enumerate(LAMBDAS):
            scores = records.get((length, value, mode))
            if scores:
                grid[row, column] = float(np.mean(np.array(scores) > THRESHOLD))
    return grid


def draw(axis, grid, title, cmap, vmin, vmax, dead=None):
    image = axis.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                        aspect="auto", interpolation="nearest")
    if dead is not None:
        for row in range(len(LENGTHS)):
            for column in range(len(LAMBDAS)):
                if not dead[row, column]:
                    continue
                axis.add_patch(
                    matplotlib.patches.Rectangle(
                        (column - 0.5, row - 0.5), 1.0, 1.0,
                        facecolor="white", edgecolor="none", zorder=1.5,
                    )
                )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    axis.tick_params(length=2.5, width=matplotlib.rcParams["axes.linewidth"])
    axis.set_xticks(range(len(LAMBDAS)))
    axis.set_xticklabels([f"{v:g}" for v in LAMBDAS])
    axis.set_yticks(range(len(LENGTHS)))
    axis.set_yticklabels([str(v) for v in LENGTHS])
    axis.set_title(title, fontsize=8)
    for row in range(len(LENGTHS)):
        for column in range(len(LAMBDAS)):
            value = grid[row, column]
            if np.isnan(value):
                continue
            if dead is not None and dead[row, column]:
                shade = "0.45"
            else:
                shade = "white" if value < (vmin + vmax) / 2 else "black"
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center",
                      fontsize=5.5, color=shade, zorder=2.0)
    return image


def main():
    width = FIG_WIDTH
    height = (width / 3) / GOLDEN_RATIO * 1.45
    figure, axes = plt.subplots(1, 3, figsize=(width, height), constrained_layout=True)

    image = None
    for axis, (name, path, label) in zip(axes, ALGORITHMS):
        if not path.exists():
            continue
        records = pickle.loads(path.read_bytes())
        truncated = matrix(records, "TBPTT(1)")
        exact = matrix(records, "RTRL")
        difference = exact - truncated
        dead = np.isclose(difference, 0.0) & (np.fmax(truncated, exact) < THRESHOLD)
        image = draw(axis, difference, label, "magma", -0.25, 1.0, dead)
        print(f"  {name}: max {np.nanmax(difference):+.2f}, "
              f"min {np.nanmin(difference):+.2f}, "
              f"mean {np.nanmean(difference):+.2f}, "
              f"dead cells {int(dead.sum())}/{dead.size}")

    axes[0].set_ylabel(r"Chain length $L$")
    for axis in axes[1:]:
        axis.set_yticklabels([])
    figure.colorbar(image, ax=axes.tolist(), fraction=0.030, pad=0.02,
                    label="RTRL $-$ TBPTT(1)")
    figure.supxlabel(r"Trace decay $\lambda$", fontsize=9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
