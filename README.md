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
`algorithm.gamma=0.95`, `num_seeds=10`, `cell.cell.config.features=64`,
`logger=[wandb]`.

## Reproducing the paper

Predefined sweeps live in `config/experiment/` and are selected with `experiment=<name>`:

| Experiment | Paper section |
| --- | --- |
| `qrc_memory_chain` | 4.1 — MemoryChain |
| `qrc_popjym`, `stream_ac_popjym`, `ppo_popjym` | 4.2 — POPGym |
| `stream_ac_brax`, `ppo_brax` | 4.3 — Masked MuJoCo |
| `qrc_k_memory_chain`, `stream_ac_k_memory_chain`, `k_memory_chain` | 4.4 — RTRL staleness |

```bash
uv run main.py experiment=qrc_popjym
uv run main.py experiment=stream_ac_brax
```

Each experiment file pins algorithm/mode/cell/logger and declares a Hydra
multirun sweep over seeds, environments, and cell variants. Benchmarks:
MemoryChain (`gymnax/bsuite/memory_chain`), five POPGym memory tasks
(`popjym/easy/*`), masked MuJoCo (`brax/*`, mask positions with
`environment.kwargs.mode=P` or velocities with `V`), and KMemoryChain
(`rsrl/k_memory_chain`).

## Repository layout

```
src/
├── algorithms/        # qrc, stream_ac, ppo; one file per (algorithm, env-family)
│   ├── eligibility_trace.py
│   ├── optimizers/    # OBGD + logging wrappers
│   └── algorithm.py   # (algorithm, env-family) -> make() registry
├── cells/             # RTU cell, RTRL wrapper
├── environments/      # KMemoryChain + observation/reward wrappers
└── utils/             # profile (SPS), resolvers, staleness buffer, initializers

config/                # hydra: algorithm, environment, cell, mode, hyperparameters, experiment, logger
main.py                # entry point
```

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
