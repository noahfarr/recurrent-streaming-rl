import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = "/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/horizon_heatmap.pdf"
LENGTHS = [16, 32, 64, 96, 128, 192, 256]
POINTS = 50


def mode_of(run):
    target = str(run.config.get("mode", {}).get("wrapper", {}).get("_target_", ""))
    return "RTRL" if target.endswith("RTRL") else "TBPTT(1)"


def collect():
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.num_seeds": 32,
            "config.total_timesteps": 500000,
            "config.environment.env_id": "MemoryChain-bsuite",
        },
        order="-created_at",
    )
    series = {}
    for run in runs:
        config = run.config
        environment = config.get("environment", {})
        if environment.get("wrappers"):
            continue
        if "MinGRU" not in str(config.get("cell", {}).get("config", {}).get("_target_", "")):
            continue
        algorithm = config.get("algorithm", {})
        if algorithm.get("gamma") != 0.999 or algorithm.get("trace_lambda") != 1.0:
            continue
        length = ((environment.get("env_params") or {}).get("max_steps_in_episode") or 1) - 1
        if length not in LENGTHS:
            continue
        history = run.history(keys=["training/episode_returns"], samples=120)
        if history.empty:
            continue
        series.setdefault((mode_of(run), length), []).append(
            (history["_step"].to_numpy(), history["training/episode_returns"].to_numpy())
        )
    return series


def surface(series, mode):
    grid = np.linspace(0, 500_000, POINTS)
    image = np.full((len(LENGTHS), POINTS), np.nan)
    for row, length in enumerate(LENGTHS):
        curves = series.get((mode, length))
        if not curves:
            continue
        values = np.stack([np.interp(grid, s, v) for s, v in curves])
        image[row] = (values > 0.5).mean(axis=0)
        print(f"  {mode:9s} L={length:4d} n={len(curves):2d} final={image[row, -1]:.0%}")
    return grid, image


def main():
    series = collect()
    grid, tbptt = surface(series, "TBPTT(1)")
    _, rtrl = surface(series, "RTRL")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    extent = [0, 500, -0.5, len(LENGTHS) - 0.5]
    common = dict(aspect="auto", origin="lower", extent=extent, interpolation="nearest")

    for axis, image, title in [
        (axes[0], tbptt, "TBPTT(1)"),
        (axes[1], rtrl, "RTRL"),
    ]:
        handle = axis.imshow(image, vmin=0, vmax=1, cmap="viridis", **common)
        axis.set_title(f"{title}: seeds solved")
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)

    difference = rtrl - tbptt
    handle = axes[2].imshow(difference, vmin=-1, vmax=1, cmap="RdBu_r", **common)
    axes[2].set_title("RTRL $-$ TBPTT(1)")
    figure.colorbar(handle, ax=axes[2], fraction=0.046, pad=0.03)

    for axis in axes:
        axis.set_yticks(range(len(LENGTHS)))
        axis.set_yticklabels(LENGTHS)
        axis.set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("chain length $L$")

    figure.tight_layout()
    figure.savefig(OUTPUT, bbox_inches="tight")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
