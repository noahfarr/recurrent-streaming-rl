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

## Method

An RTU layer trained by exact real-time recurrent learning (RTRL) is inserted
between the observation and the feedforward head of an existing streaming RL
algorithm. The streaming update machinery — eligibility traces and step-size
adaptation — is left unchanged: the RTRL trace and the eligibility trace compose
without modification, yielding a single-pass procedure that propagates credit
through both the recurrent state and time, with no truncation and no replay
buffer.

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

> This branch (`streax-port`) reproduces the paper's algorithms on top of
> [`streax`](https://github.com/noahfarr/streax) instead of hand-rolled
> eligibility-trace/optimizer/RTU-RTRL code, and covers §4.1–4.3 only. §4.4
> (RTRL staleness, KMemoryChain) stays on `main`, which still uses the
> original implementation.

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
(`popgymnax/easy/*`), and masked MuJoCo (`brax/*`, mask velocities with
`environment.kwargs.mode=P` or positions with `V`).

## Repository layout

```
src/
├── algorithms/        # algorithm.py: (algorithm, env-family) -> make() registry
│   ├── optimizers/    # logging wrapper for PPO's optax optimizers
│   └── ppo/           # batched PPO baseline (streax Network/RTU/RTRL, own training loop)
├── recipes/           # one make() factory per (algorithm, env-family): networks + wrappers + agent
├── cells/             # RTU cell, RNN wrapper (TBPTT), RTRL wrapper (exact online gradient)
├── networks/          # Network composition, feature-extractor/head building blocks
├── environments/      # observation/reward/action wrappers
└── utils/             # profile (SPS), resolvers, typing, initializers

config/                # hydra: algorithm, environment, cell, mode, hyperparameters, experiment, logger
main.py                # entry point
```

`qrc` and `stream_ac` are thin factories around `streax.algorithms.RecurrentQRCLambda`
and `RecurrentACLambda`; the algorithm math itself lives in `streax`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{farr2026streaming,
  title   = {Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning},
  author  = {Farr, Noah and Reddi, Aryaman and D'Eramo, Carlo and Peters, Jan},
  year    = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
