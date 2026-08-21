import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import lox
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import algorithm

COMBOS = [
    ("ffn", "bptt", None, 8192),
    ("gru", "bptt", 64, 8192),
    ("gru", "bptt", 128, 8192),
    ("gru", "bptt", 256, 8192),
    ("gru", "bptt", 512, 8192),
    ("gru", "rtrl", 64, 512),
    ("gru", "rtrl", 128, 128),
    ("gru", "rtrl", 256, 16),
    ("gru", "snap1", 64, 2048),
    ("gru", "snap1", 128, 2048),
    ("gru", "snap1", 256, 1024),
    ("gru", "snap1", 512, 512),
    ("gru", "uoro", 64, 2048),
    ("gru", "uoro", 128, 2048),
    ("gru", "uoro", 256, 1024),
    ("gru", "uoro", 512, 512),
    ("min_gru", "bptt", 64, 8192),
    ("min_gru", "bptt", 256, 8192),
    ("min_gru", "rtrl", 64, 4096),
    ("min_gru", "rtrl", 128, 4096),
    ("min_gru", "rtrl", 256, 2048),
    ("min_gru", "rtrl", 512, 1024),
    ("rtu", "bptt", 64, 8192),
    ("rtu", "bptt", 256, 8192),
    ("rtu", "rtrl", 64, 4096),
    ("rtu", "rtrl", 128, 4096),
    ("rtu", "rtrl", 256, 2048),
    ("rtu", "rtrl", 512, 1024),
]

SEED_COMBOS = [
    ("ffn", "bptt", None),
    ("gru", "bptt", 256),
    ("min_gru", "rtrl", 256),
    ("rtu", "rtrl", 256),
]

SEED_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SEED_FIELDS = [
    "cell",
    "mode",
    "hidden",
    "backend",
    "num_seeds",
    "num_steps",
    "reps",
    "compile_s",
    "total_sps",
    "per_seed_sps",
    "error",
]

FIELDS = [
    "cell",
    "mode",
    "hidden",
    "backend",
    "num_steps",
    "reps",
    "actor_param_count",
    "actor_carry_size",
    "compile_full_s",
    "full_ms",
    "sps",
    "act_ms",
    "env_ms",
    "spool_ms",
    "update_ms",
    "error",
]


def make_cfg(cell, mode, hidden):
    overrides = [
        "algorithm=stream_ac",
        "environment=popgymnax/repeat_first/easy",
        f"cell={cell}",
        f"mode={mode}",
    ]
    if hidden is not None:
        if cell == "gru":
            overrides.append(f"cell.features={hidden}")
        else:
            overrides.append(f"cell.config.hidden_dim={hidden}")
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def build_fns(agent, num_steps):
    def full(key, state):
        return agent.train(key, state, num_steps)

    def act(key, state):
        def step(carry, key):
            state, acc = carry
            state, transition = agent.env_step(state, key, 1.0)
            grads = (
                transition.aux["log_prob_grads"],
                transition.aux["entropy_grads"],
                transition.aux["critic_grads"],
            )
            acc = acc + sum(jnp.sum(leaf) for leaf in jax.tree.leaves(grads))
            return (state, acc), None

        (state, acc), _ = jax.lax.scan(
            step,
            (state, jnp.float32(0.0)),
            jax.random.split(key, num_steps),
            unroll=agent.cfg.unroll,
        )
        return acc

    def env_only(key, state):
        action = state.timestep.action

        def step(carry, key):
            env_state, acc = carry
            obs, env_state, reward, done, info = agent.env.step(
                key, env_state, action, agent.env_params
            )
            acc = acc + reward + jnp.sum(obs)
            return (env_state, acc), None

        (env_state, acc), _ = jax.lax.scan(
            step,
            (state.env_state, jnp.float32(0.0)),
            jax.random.split(key, num_steps),
            unroll=agent.cfg.unroll,
        )
        return acc

    spooled = lox.spool(agent.train)

    def spool(key, state):
        return spooled(key, state, num_steps)

    return {"full": full, "act": act, "env": env_only, "spool": spool}


def timed(fn, key, state, reps):
    start = time.perf_counter()
    jax.block_until_ready(fn(key, state))
    compile_s = time.perf_counter() - start
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(key, state))
        times.append(time.perf_counter() - start)
    return compile_s, sorted(times)[len(times) // 2]


def run_combo(cell, mode, hidden, num_steps, reps):
    cfg = make_cfg(cell, mode, hidden)
    agent = algorithm.make(cfg)

    key = jax.random.key(0)
    init_key, run_key = jax.random.split(key)
    state = jax.jit(agent.init)(init_key)
    jax.block_until_ready(state)

    row = {
        "cell": cell,
        "mode": mode,
        "hidden": hidden,
        "backend": jax.default_backend(),
        "num_steps": num_steps,
        "reps": reps,
        "actor_param_count": sum(
            leaf.size for leaf in jax.tree.leaves(state.params)
        ),
        "actor_carry_size": sum(
            leaf.size for leaf in jax.tree.leaves(state.carry)
        ),
    }

    fns = build_fns(agent, num_steps)
    errors = []
    for name, fn in fns.items():
        try:
            compile_s, median_s = timed(jax.jit(fn), run_key, state, reps)
        except Exception as e:
            errors.append(f"{name}: {e!r}"[:120])
            continue
        row[f"{name}_ms"] = median_s / num_steps * 1e3
        if name == "full":
            row["compile_full_s"] = compile_s
            row["sps"] = num_steps / median_s

    if "full_ms" in row and "act_ms" in row:
        row["update_ms"] = row["full_ms"] - row["act_ms"]
    if errors:
        row["error"] = "; ".join(errors)
    return row


def run_seed_combo(cell, mode, hidden, num_seeds, num_steps, reps):
    cfg = make_cfg(cell, mode, hidden)
    agent = algorithm.make(cfg)

    init_keys = jax.random.split(jax.random.key(0), num_seeds)
    state = jax.jit(jax.vmap(agent.init))(init_keys)
    jax.block_until_ready(state)

    train = jax.vmap(agent.train, in_axes=(0, 0, None))

    def fn(key, state):
        return train(jax.random.split(key, num_seeds), state, num_steps)

    compile_s, median_s = timed(jax.jit(fn), jax.random.key(1), state, reps)
    return {
        "cell": cell,
        "mode": mode,
        "hidden": hidden,
        "backend": jax.default_backend(),
        "num_seeds": num_seeds,
        "num_steps": num_steps,
        "reps": reps,
        "compile_s": compile_s,
        "total_sps": num_seeds * num_steps / median_s,
        "per_seed_sps": num_steps / median_s,
    }


def run_seed_scaling(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_FIELDS)
        if write_header:
            writer.writeheader()
        for cell, mode, hidden in SEED_COMBOS:
            for num_seeds in SEED_COUNTS:
                label = f"{cell}-{mode}-{hidden}-s{num_seeds}"
                if args.only and args.only not in label:
                    continue
                try:
                    row = run_seed_combo(
                        cell, mode, hidden, num_seeds, args.num_steps, args.reps
                    )
                except Exception as e:
                    row = {
                        "cell": cell,
                        "mode": mode,
                        "hidden": hidden,
                        "backend": jax.default_backend(),
                        "num_seeds": num_seeds,
                        "num_steps": args.num_steps,
                        "reps": args.reps,
                        "error": repr(e)[:200],
                    }
                    print(f"{label}: ERROR {e!r}", flush=True)
                writer.writerow(row)
                f.flush()
                if "total_sps" in row:
                    print(
                        f"{label}: total_SPS={row['total_sps']:,.0f} "
                        f"per_seed={row['per_seed_sps']:,.0f} "
                        f"compile={row['compile_s']:.1f}s",
                        flush=True,
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--only", default=None)
    parser.add_argument("--seed-scaling", action="store_true")
    parser.add_argument("--num-steps", type=int, default=1024)
    args = parser.parse_args()

    if args.seed_scaling:
        args.output = args.output or str(REPO / "benchmarks" / "seed_scaling.csv")
        run_seed_scaling(args)
        return

    output = Path(args.output or str(REPO / "benchmarks" / "sps.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)

    combos = COMBOS
    if args.only:
        combos = [c for c in combos if args.only in f"{c[0]}-{c[1]}-{c[2]}"]

    write_header = not output.exists()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for cell, mode, hidden, num_steps in combos:
            label = f"{cell}-{mode}-{hidden}"
            start = time.perf_counter()
            try:
                row = run_combo(cell, mode, hidden, num_steps, args.reps)
            except Exception as e:
                row = {
                    "cell": cell,
                    "mode": mode,
                    "hidden": hidden,
                    "backend": jax.default_backend(),
                    "num_steps": num_steps,
                    "reps": args.reps,
                    "error": repr(e)[:200],
                }
                print(f"{label}: ERROR {e!r}", flush=True)
            writer.writerow(row)
            f.flush()
            if "sps" in row:
                print(
                    f"{label}: SPS={row['sps']:,.0f} full={row['full_ms']:.3f}ms "
                    f"act={row['act_ms']:.3f}ms env={row['env_ms']:.3f}ms "
                    f"spool={row['spool_ms']:.3f}ms compile={row['compile_full_s']:.1f}s "
                    f"({time.perf_counter() - start:.0f}s total)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
