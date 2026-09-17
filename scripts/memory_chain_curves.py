import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT = "noahfarr/recurrent-streaming-rl"
OUTPUT = "/home/farr/recurrent-streaming-rl/paper/plots/memory_chain/tbptt_vs_rtrl.pdf"


def mode_of(run):
    target = str(run.config.get("mode", {}).get("wrapper", {}).get("_target_", ""))
    return "RTRL" if target.endswith("RTRL") else "TBPTT(1)"


def collect(length, trace_lambda):
    api = wandb.Api()
    runs = api.runs(
        PROJECT,
        filters={
            "state": "finished",
            "config.num_seeds": 16,
            "config.total_timesteps": 500000,
            "config.environment.env_id": "MemoryChain-bsuite",
        },
        order="-created_at",
    )
    series = {"RTRL": [], "TBPTT(1)": []}
    for run in list(runs)[:200]:
        config = run.config
        environment = config.get("environment", {})
        if environment.get("wrappers"):
            continue
        cell = str(config.get("cell", {}).get("config", {}).get("_target_", ""))
        if "MinGRU" not in cell:
            continue
        algorithm = config.get("algorithm", {})
        if algorithm.get("gamma") != 0.999:
            continue
        if abs((algorithm.get("trace_lambda") or 0) - trace_lambda) > 1e-9:
            continue
        params = environment.get("env_params") or {}
        if (params.get("max_steps_in_episode") or 1) - 1 != length:
            continue
        history = run.history(keys=["training/episode_returns"], samples=120)
        if history.empty:
            continue
        series[mode_of(run)].append(
            (history["_step"].to_numpy(), history["training/episode_returns"].to_numpy())
        )
    return series


def grid_align(curves, points=60):
    limit = min(c[0].max() for c in curves)
    grid = np.linspace(0, limit, points)
    return grid, np.stack([np.interp(grid, s, v) for s, v in curves])


def main():
    length, trace_lambda = 128, 1.0
    series = collect(length, trace_lambda)
    colours = {"TBPTT(1)": "#c2453a", "RTRL": "#2f5d8a"}

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, curves in series.items():
        if not curves:
            continue
        grid, values = grid_align(curves)
        for row in values:
            axes[0].plot(grid / 1e3, row, color=colours[name], alpha=0.22, linewidth=0.7)
        median = np.median(values, axis=0)
        axes[0].plot(grid / 1e3, median, color=colours[name], linewidth=2.2, label=f"{name} (median)")
        solved = (values > 0.5).mean(axis=0)
        axes[1].plot(grid / 1e3, 100 * solved, color=colours[name], linewidth=2.2, label=name)
        print(f"{name:9s} n={len(curves):2d} final median={median[-1]:+.3f} solve={100 * solved[-1]:.0f}%")

    axes[0].set_xlabel("environment steps (thousands)")
    axes[0].set_ylabel("episodic return")
    axes[0].set_title(f"MemoryChain $L={length}$, $\\lambda={trace_lambda:g}$, minGRU")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    axes[0].set_ylim(-0.35, 1.05)
    axes[1].set_ylim(-2, 102)
    axes[1].set_xlabel("environment steps (thousands)")
    axes[1].set_ylabel("seeds solved (%)")
    axes[1].set_title("fraction of seeds above 0.5 return")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT, bbox_inches="tight")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
