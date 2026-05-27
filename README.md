# Recurrent Streaming Reinforcement Learning

JAX implementation of the experiments in *Streaming Reinforcement Learning under
Partial Observability with Real-Time Recurrent Learning*. A Recurrent Trace Unit
(RTU) layer trained by exact real-time recurrent learning (RTRL) is inserted
between the observation and the feedforward head of a streaming RL algorithm,
propagating credit through both the recurrent state and time without truncation.

## Install

```bash
uv sync
```

## Algorithms

`config/algorithm/`:

- `qrc` — QRC(λ), value-based streaming control.
- `stream_ac` — stream AC(λ), policy-based streaming control.
- `ppo` — batched PPO baseline (relaxes the streaming constraint with a replay/rollout buffer).

## Recurrent cells and credit assignment

`config/cell/`: `rtu` (Recurrent Trace Unit), `gru`, `ffn` (feedforward identity).

`config/mode/` toggles how the cell is credited:

- `bptt` — truncated backpropagation through time (TBPTT(1) under streaming).
- `rtrl` — wraps the cell with `src.cells.RTRL` for an exact online gradient.

## Environments

`config/environment/`:

- `gymnax/bsuite/memory_chain` — MemoryChain diagnostic (Section 4.1).
- `popjym/easy/{autoencode,concentration,count_recall,higher_lower,repeat_first}` — five POPGym memory tasks (Section 4.2).
- `brax/{ant,halfcheetah,hopper,walker2d}` — masked MuJoCo continuous control; mask positions (`environment.kwargs.mode=P`) or velocities (`V`) (Section 4.3).
- `rsrl/k_memory_chain` — KMemoryChain, an every-step memory variant used for the RTRL staleness analysis (Section 4.4).

## Run a single experiment

```bash
uv run main.py algorithm=qrc environment=gymnax/bsuite/memory_chain cell=rtu mode=rtrl
```

Algorithm-specific hyperparameters are picked up automatically via the
`cascading_fallback` resolver from `config/hyperparameters/`. Override anything at
the CLI, e.g. `algorithm.gamma=0.95`, `num_seeds=10`,
`cell.cell.config.features=64`, `logger=[wandb]`.

## Paper experiments

Predefined sweeps live in `config/experiment/`, selected with `experiment=<name>`:

| Experiment | Paper section |
| --- | --- |
| `qrc_memory_chain` | 4.1 MemoryChain |
| `qrc_popjym`, `stream_ac_popjym`, `ppo_popjym` | 4.2 POPGym |
| `stream_ac_brax`, `ppo_brax` | 4.3 Masked MuJoCo |
| `qrc_k_memory_chain`, `stream_ac_k_memory_chain`, `k_memory_chain` | 4.4 RTRL staleness |

```bash
uv run main.py experiment=qrc_popjym
uv run main.py experiment=stream_ac_brax
```

Each experiment pins algorithm/mode/cell/logger and declares a Hydra multirun
sweep over seeds, environments, and cell variants.

## Layout

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
