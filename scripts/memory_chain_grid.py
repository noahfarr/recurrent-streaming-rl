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
from styles import FIG_WIDTH, GOLDEN_RATIO, apply as apply_style

apply_style()

PROJECT = "noahfarr/recurrent-streaming-rl"
ALGORITHM = next((a for a in sys.argv[1:] if not a.startswith("-")), "qrc")
SUFFIX = "" if ALGORITHM == "qrc" else f"_{ALGORITHM}"
OUTPUT = Path(
    f"/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/horizon_grid{SUFFIX}.pdf"
)
CACHE = Path(
    f"/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/.grid{SUFFIX}.pkl"
)
LENGTHS = [16, 32, 64, 96, 128, 192, 256]
LAMBDAS = [0.8, 0.9, 0.98, 0.995, 1.0]
MODES = ["TBPTT(1)", "RTRL"]
GAMMA = 0.999
THRESHOLD = 0.5
SEED_CAP = 32
# The 2026-09-06 batch contradicts the August runs at identical logged config (e.g. lambda=1,
# L=32, TBPTT(1): August solves 1.00, September 0.00). Where both exist we take August; cells
# the August sweep never covered (lambda 0.8 and 0.9) fall back to September.
PREFER_MONTH = "2026-08"


def mode_of(config):
    target = str((config.get("mode") or {}).get("wrapper", {}).get("_target_", ""))
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
            "displayName": {"$regex": f"^{ALGORITHM}"},
        },
        order="-created_at",
    )
    preferred, fallback = defaultdict(list), defaultdict(list)
    for run in runs:
        config = run.config
        environment = config.get("environment", {})
        algorithm = config.get("algorithm", {})
        if environment.get("wrappers"):
            continue
        if "MinGRU" not in str(config.get("cell", {}).get("config", {}).get("_target_", "")):
            continue
        if algorithm.get("gamma") != GAMMA:
            continue
        if not run.name.startswith(ALGORITHM):
            continue
        score = run.summary.get("score")
        if score is None:
            continue
        length = ((environment.get("env_params") or {}).get("max_steps_in_episode") or 1) - 1
        key = (length, algorithm.get("trace_lambda"), mode_of(config))
        bucket = preferred if str(run.created_at)[:7] == PREFER_MONTH else fallback
        if len(bucket[key]) >= SEED_CAP:
            continue
        bucket[key].append(score)
    records = {k: v for k, v in fallback.items()}
    records.update({k: v for k, v in preferred.items() if v})
    CACHE.write_bytes(pickle.dumps(records))
    return records


def matrix(records, mode):
    grid = np.full((len(LENGTHS), len(LAMBDAS)), np.nan)
    for row, length in enumerate(LENGTHS):
        for column, value in enumerate(LAMBDAS):
            scores = records.get((length, value, mode), [])
            if scores:
                grid[row, column] = np.mean([s > THRESHOLD for s in scores])
    return grid


def draw(axis, grid, title, cmap, vmin, vmax):
    image = axis.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="auto",
                        interpolation="nearest")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    axis.tick_params(length=2.5, width=matplotlib.rcParams["axes.linewidth"])
    axis.set_xticks(range(len(LAMBDAS)))
    axis.set_xticklabels([f"{v:g}" for v in LAMBDAS])
    axis.set_yticks(range(len(LENGTHS)))
    axis.set_yticklabels([str(v) for v in LENGTHS])
    axis.set_xlabel(r"Trace decay $\lambda$")
    axis.set_title(title, fontsize=8)
    for row in range(len(LENGTHS)):
        for column in range(len(LAMBDAS)):
            value = grid[row, column]
            if np.isnan(value):
                continue
            shade = "white" if value < (vmin + vmax) / 2 else "black"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      fontsize=5.5, color=shade)
    return image


def boundary_cells(axis, grid):
    width = matplotlib.rcParams["axes.linewidth"]
    inset_x = inset_y = 0.0
    for column in range(len(LAMBDAS)):
        solved = [r for r in range(len(LENGTHS)) if grid[r, column] > 0.5]
        if not solved:
            continue
        row = max(solved)
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (column - 0.5 + inset_x, row - 0.5 + inset_y),
                1.0 - 2 * inset_x,
                1.0 - 2 * inset_y,
                fill=False,
                edgecolor="black",
                linewidth=width,
                joinstyle="miter",
                clip_on=False,
                snap=False,
                zorder=2.0,
            )
        )


def main():
    records = fetch()
    truncated = matrix(records, "TBPTT(1)")
    exact = matrix(records, "RTRL")

    width = FIG_WIDTH
    height = (width / 3) / GOLDEN_RATIO * 1.35
    figure, axes = plt.subplots(1, 3, figsize=(width, height), constrained_layout=True)

    draw(axes[0], truncated, "TBPTT(1)", "viridis", 0.0, 1.0)
    image = draw(axes[1], exact, "RTRL", "viridis", 0.0, 1.0)
    for axis, grid in ((axes[0], truncated), (axes[1], exact)):
        boundary_cells(axis, grid)
    difference = draw(axes[2], exact - truncated, "RTRL $-$ TBPTT(1)", "magma", 0.0, 1.0)

    axes[0].set_ylabel(r"Chain length $L$")
    for axis in axes[1:]:
        axis.set_yticklabels([])
    figure.colorbar(image, ax=axes[:2].tolist(), fraction=0.035, pad=0.03, label="Success rate")
    figure.colorbar(
        difference, ax=axes[2], fraction=0.06, pad=0.03, label="Success rate"
    )
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")
    for row, length in enumerate(LENGTHS):
        print(f"  L={length:<4}" + "".join(
            f"{truncated[row, c]:5.2f}/{exact[row, c]:<5.2f}" for c in range(len(LAMBDAS))))


if __name__ == "__main__":
    main()
