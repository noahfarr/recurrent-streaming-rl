# Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Official JAX implementation of the paper [**Streaming Reinforcement Learning
under Partial Observability with Real-Time Recurrent Learning**](paper.pdf) by
Noah Farr, Aryaman Reddi, Carlo D'Eramo, and Jan Peters.

Technical University of Darmstadt · University of Würzburg · Hessian.AI · DFKI · Zuse School ELIZA</sub>

## Abstract

Streaming reinforcement learning has emerged as an online learning paradigm that
conforms to the restrictions of natural learning agents that process data
incrementally, i.e. with a batch size of 1 and no replay buffer. While streaming
RL has recently been shown to scale with deep function approximation under full
observability, partially observable settings have remained out of reach.
Truncated backpropagation through time collapses to a one-step gradient horizon
under the streaming setting, and exact real-time recurrent learning is
prohibitively expensive. We close this gap using **Recurrent Trace Units
(RTUs)**, a diagonal recurrent architecture that enables exact RTRL with linear
time and memory complexity in the parameter count, and show that they integrate
cleanly into existing streaming algorithms across both discrete and continuous
control. On a MemoryChain diagnostic with chain lengths from 2 to 128, our method
sustains performance where streaming TBPTT(1) baselines using feedforward, GRU,
and RTU networks collapse. On five POPGym tasks and on partially observable
MuJoCo continuous control, the streaming approach is competitive with batched PPO
on POPGym and recovers a substantial fraction of batched performance on masked
MuJoCo, despite using no replay buffer or batched updates.

## Installation

```bash
uv sync
```

## Quickstart

Run a single configuration (algorithm × environment × cell × credit-assignment mode):

```bash
uv run main.py algorithm=qrc environment=gymnax/bsuite/memory_chain cell=rtu mode=rtrl
```

- **Algorithms** (`config/algorithm/`): `qrc` — QRC(λ); `stream_ac` — stream AC(λ); `ppo` — batched PPO baseline.
- **Cells** (`config/cell/`): `rtu` (Recurrent Trace Unit), `gru`, `ffn` (feedforward identity).
- **Modes** (`config/mode/`): `bptt` (TBPTT(1) under streaming) and `rtrl` (exact online gradient via `src.cells.RTRL`).

Algorithm-specific hyperparameters are resolved automatically from
`config/hyperparameters/`. Override anything at the CLI, e.g.
`algorithm.gamma=0.95`, `num_seeds=10`, `cell.config.features=64`,
`logger=[wandb]`.

## Reproducing the paper

Predefined sweeps live in `config/experiment/` and are selected with `experiment=<name>`:

| Experiment | Paper section |
| --- | --- |
| `qrc_memory_chain` | 4.1 — MemoryChain |
| `qrc_popgymnax`, `stream_ac_popgymnax`, `ppo_popgymnax` | 4.2 — POPGym |
| `stream_ac_brax`, `ppo_brax` | 4.3 — Masked MuJoCo |

```bash
uv run main.py experiment=qrc_popgymnax
uv run main.py experiment=stream_ac_brax
```

Each experiment file pins algorithm/mode/cell/logger and declares a Hydra
multirun sweep over seeds, environments, and cell variants. Benchmarks:
MemoryChain (`gymnax/bsuite/memory_chain`), five POPGym memory tasks
(`popgymnax/*/easy`), and masked MuJoCo (`brax/*`, mask velocities with
`environment.kwargs.mode=P` or positions with `V`).

## Hyperparameter sweep

Hydra multirun (above) runs a fixed grid. For actual hyperparameter tuning,
`scripts/sweep_hyperparameters.py` runs a
[CARBS](https://github.com/imbue-ai/carbs) (cost-aware Bayesian optimization)
loop that submits each trial as a `main.py` subprocess via
[submitit](https://github.com/facebookincubator/submitit), reading back the
`{score, cost}` pair each run writes to `result.json`:

```bash
uv sync --extra sweeps  # installs carbs, torch (cpu), submitit
uv run python scripts/sweep_hyperparameters.py algorithm=qrc_lambda cell=rtu mode=rtrl
```

The search space and optimizer settings for a given `(algorithm, environment)`
pair live under `config/sweep/`, resolved with the same cascading fallback as
`config/hyperparameters/`, e.g. `config/sweep/qrc_lambda/popgymnax.yaml`.
Trials, the running best (`optimization_results.yaml`), and raw observations
(`observations.jsonl`) are written under `sweeps/<algorithm>/<environment>/<timestamp>/`.
Set `sweep.cluster=slurm` (and tune `sweep.executor`) to submit trials to a
Slurm cluster instead of running them locally. Each trial also runs with the
wandb logger enabled and is tagged with a `sweep` config field (the sweep's
own output directory name) so all of a sweep's trials can be pulled back
from wandb as a unit; `group` is left at its usual meaning (algorithm +
environment), not repurposed for sweep identity.

`scripts/visualize_hyperparameters.py` is an interactive
[raylib](https://github.com/electronstudio/raylib-python-cffi) GUI for
inspecting a sweep's trials: scatter/parallel-coordinates/3D views over
score, cost, and hyperparameters, per-parameter score correlation,
score-over-time, and (when reading local files) a PCA-projected GP surrogate
heatmap. It reads either a local `observations.jsonl` or, useful when trials
ran on a cluster, a wandb sweep directly via the API (no surrogate view in
that mode, since it needs the local CARBS search-space config):

```bash
uv run python scripts/visualize_hyperparameters.py sweeps/qrc_lambda/popgymnax/<timestamp>
uv run python scripts/visualize_hyperparameters.py --wandb-sweep <timestamp>
```

## Repository layout

```
src/
├── algorithms/        # algorithm.py: (algorithm, env-family) -> make() registry
│   ├── optimizers/    # logging wrapper for PPO's optax optimizers
│   └── ppo/           # batched PPO baseline (streamlet Network/RTU/RTRL, own training loop)
├── recipes/           # one make() factory per (algorithm, env-family): networks + wrappers + agent
├── cells/             # RTU cell, RNN wrapper (TBPTT), RTRL wrapper (exact online gradient)
├── networks/          # Network composition, feature-extractor/head building blocks
├── environments/      # observation/reward/action wrappers
└── utils/             # profile (SPS), resolvers, typing, initializers, carbs (sweep search-space math)

config/                # hydra: algorithm, environment, cell, mode, hyperparameters, sweep, experiment, logger
scripts/               # sweep_hyperparameters.py: CARBS+submitit hyperparameter sweep
main.py                # entry point
```

`qrc` and `stream_ac` are thin factories around `streamlet.algorithms.RecurrentQRCLambda`
and `RecurrentACLambda`; the algorithm math itself lives in `streamlet`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{farr2026streaming,
  title   = {Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning},
  author  = {Farr, Noah and Reddi, Aryaman and D'Eramo, Carlo and Peters, Jan},
  year    = {2026}
}
```
