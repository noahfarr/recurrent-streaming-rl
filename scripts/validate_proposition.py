import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import lox
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from src import algorithm

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
LAGS = [1, 2, 4, 8, 16, 32]
ROLLOUTS = 10
EPSILONS = [0.01, 0.1]
SWEEP_ROLLOUTS = 3


def build(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return algorithm.make(cfg)


def find_param(tree, name):
    if hasattr(tree, "items"):
        for key, value in tree.items():
            if key == name:
                return value
            found = find_param(value, name)
            if found is not None:
                return found
    return None


def rtu_retention(network, params, hidden, timestep):
    jacobian = jax.jacrev(lambda h: network.apply(params, h, *timestep)[0])(hidden)
    real = np.diag(np.asarray(jacobian.real.real))
    imaginary = np.diag(np.asarray(jacobian.imaginary.real))
    return real + 1j * imaginary


def fitted_rate(values, lags):
    mask = (values > 1e-12) & (lags > 0) & (lags < lags.max())
    if mask.sum() < 5:
        return float("nan")
    slope = np.polyfit(lags[mask], np.log(values[mask]), 1)[0]
    return float(np.exp(slope))


def carry_of(state):
    return state.q_carry if hasattr(state, "q_carry") else state.carry


def collect(env_step, network, params, state, key, length, cell, lam, epsilon=0.01):
    heads, retentions, cuts = [], [], []
    for _ in range(length + 1):
        hidden, timestep = carry_of(state).carry, state.timestep
        head = jax.grad(lambda h: network.apply(params, h, *timestep)[1].max())(hidden)
        if cell == "rtu":
            heads.append(
                np.asarray(head.real).ravel() + 1j * np.asarray(head.imaginary).ravel()
            )
            retentions.append(lam)
        else:
            jacobian = jax.jacrev(lambda h: network.apply(params, h, *timestep)[0])(hidden)
            flat = jnp.reshape(jacobian, (hidden.size, hidden.size))
            heads.append(np.asarray(head).ravel())
            retentions.append(np.asarray(jnp.diag(flat)))
        key, step_key = jax.random.split(key)
        state, transition = env_step(state, step_key, epsilon)
        cuts.append(
            bool(transition.aux["non_greedy"]) or bool(transition.second.done)
        )
    return np.stack(heads), np.stack(retentions), np.array(cuts), state, key


def statistics(heads, retentions, decay, cuts=None):
    horizon = len(heads)
    survivor = 0
    if cuts is not None:
        indices = [i for i in range(horizon) if cuts[i]]
        survivor = indices[-1] + 1 if indices else 0

    truncated_weights, exact_weights, incoherent_weights = [], [], []
    for k in range(horizon):
        if k < survivor:
            truncated_weights.append(0.0)
        else:
            truncated_weights.append(
                np.linalg.norm(decay ** (horizon - 1 - k) * heads[k])
            )
        accumulated = np.zeros_like(heads[0])
        magnitudes = np.zeros(heads[0].shape, dtype=float)
        product = np.ones_like(heads[0])
        for i in range(k, horizon):
            if i > k:
                product = product * retentions[i]
            if i < survivor:
                continue
            term = decay ** (horizon - 1 - i) * np.conj(heads[i]) * product
            accumulated = accumulated + term
            magnitudes = magnitudes + np.abs(term)
        exact_weights.append(np.linalg.norm(accumulated))
        incoherent_weights.append(np.linalg.norm(magnitudes))

    truncated_weights = np.array(truncated_weights)
    exact_weights = np.array(exact_weights)
    incoherent_weights = np.array(incoherent_weights)
    lags = np.arange(horizon - 1, -1, -1)

    ratios, coherence, destroyed = {}, {}, {}
    for lag in LAGS:
        if lag < horizon:
            index = horizon - 1 - lag
            denominator = truncated_weights[index]
            ratios[lag] = (
                exact_weights[index] / denominator
                if denominator > 1e-12
                else float("nan")
            )
            spread = incoherent_weights[index]
            coherence[lag] = (
                exact_weights[index] / spread if spread > 1e-12 else float("nan")
            )
            destroyed[lag] = float(
                denominator <= 1e-12 and exact_weights[index] > 1e-12
            )

    return {
        "rho": float(np.max(np.abs(retentions))),
        "tbptt": fitted_rate(truncated_weights, lags),
        "rtrl": fitted_rate(exact_weights, lags),
        "ratios": ratios,
        "coherence": coherence,
        "destroyed": destroyed,
    }


def middle(values):
    values = np.array([v for v in values if np.isfinite(v)])
    return float(np.median(values)) if values.size else float("nan")


def measure(agent, truncated, length, steps, decay, seed, cell, epsilon):
    keys = jax.random.split(jax.random.key(seed), 8)
    state = jax.jit(agent.init)(keys[0])
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    for index in range(6):
        state, _ = train(keys[index + 1], state, max(steps // 6, 1))

    params = state.params
    network = truncated.q_network
    env_step = jax.jit(agent.env_step)
    lam = None
    if cell == "rtu":
        lam = rtu_retention(
            network, params, carry_of(state).carry, state.timestep
        )

    key = keys[7]
    samples, cut_samples, fractions = [], [], []
    for _ in range(ROLLOUTS):
        heads, retentions, cuts, state, key = collect(
            env_step, network, params, state, key, length, cell, lam, epsilon
        )
        samples.append(statistics(heads, retentions, decay))
        cut_samples.append(statistics(heads, retentions, decay, cuts))
        fractions.append(float(np.mean(cuts)))

    sweep = {value: {"destroyed": [], "fraction": []} for value in EPSILONS}
    for value in EPSILONS:
        for _ in range(SWEEP_ROLLOUTS):
            heads, retentions, cuts, state, key = collect(
                env_step, network, params, state, key, length, cell, lam, value
            )
            sweep[value]["destroyed"].append(
                statistics(heads, retentions, decay, cuts)["destroyed"]
            )
            sweep[value]["fraction"].append(float(np.mean(cuts)))

    ratios, coherence, cut_ratios, destroyed = {}, {}, {}, {}
    for lag in LAGS:
        ratios[lag] = middle([s["ratios"][lag] for s in samples if lag in s["ratios"]])
        coherence[lag] = middle(
            [s["coherence"][lag] for s in samples if lag in s["coherence"]]
        )
        cut_ratios[lag] = middle(
            [s["ratios"][lag] for s in cut_samples if lag in s["ratios"]]
        )
        destroyed[lag] = float(
            np.mean([s["destroyed"][lag] for s in cut_samples if lag in s["destroyed"]])
        )

    return {
        "rho": middle([s["rho"] for s in samples]),
        "tbptt": middle([s["tbptt"] for s in samples]),
        "rtrl": middle([s["rtrl"] for s in samples]),
        "ratios": ratios,
        "coherence": coherence,
        "cut_ratios": cut_ratios,
        "destroyed": destroyed,
        "cut_fraction": middle(fractions),
        "sweep": {
            value: {
                "destroyed": {
                    lag: float(np.mean([d[lag] for d in sweep[value]["destroyed"] if lag in d]))
                    for lag in LAGS
                },
                "fraction": middle(sweep[value]["fraction"]),
            }
            for value in EPSILONS
        },
    }


def summarise(name, values, predicted=None):
    values = np.array([v for v in values if np.isfinite(v)])
    low, high = np.percentile(values, [25, 75])
    line = f"  {name:<28} {np.median(values):>9.5f}  [{low:.5f}, {high:.5f}]"
    if predicted is not None:
        line += f"   predicted {predicted:.5f}"
    print(line, flush=True)


def main():
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    trace_lambda = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    num_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    cell = sys.argv[5] if len(sys.argv) > 5 else "min_gru"
    name = sys.argv[6] if len(sys.argv) > 6 else "qrc"
    epsilon = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
    gamma = 0.999
    decay = gamma * trace_lambda

    base = [
        f"algorithm={name}",
        "environment=gymnax/bsuite/memory_chain",
        f"environment.env_params.max_steps_in_episode={length + 1}",
        f"cell={cell}",
        f"algorithm.trace_lambda={trace_lambda}",
        f"algorithm.gamma={gamma}",
        "num_seeds=1",
        f"total_timesteps={steps}",
        "num_epochs=1",
    ]
    if name == "intentional_q":
        base = base + ["q_optimizer.cfg.eta=0.1"]
    base = base + list(sys.argv[8:])
    agent = build(base + ["mode=rtrl"])
    truncated = build(base + ["mode=bptt"])

    results = []
    for seed in range(num_seeds):
        result = measure(agent, truncated, length, steps, decay, seed, cell, epsilon)
        results.append(result)
        print(
            f"  seed {seed:>2}  rho={result['rho']:.5f}  "
            f"TBPTT={result['tbptt']:.5f}  RTRL={result['rtrl']:.5f}  "
            f"ratio@32={result['ratios'].get(32, float('nan')):.2f}",
            flush=True,
        )

    rho = np.median([r["rho"] for r in results])
    print(f"\n  L={length}, lambda={trace_lambda}, cell={cell}, algorithm={name}, {num_seeds} seeds, {steps} steps")
    print(f"  {'quantity':<28} {'median':>9}  {'[p25, p75]':>20}")
    summarise("rho (max_t ||J_t||)", [r["rho"] for r in results])
    summarise("fitted TBPTT(1)", [r["tbptt"] for r in results], decay)
    summarise("fitted RTRL", [r["rtrl"] for r in results], max(decay, rho))

    print(f"\n  cut fraction: {np.median([r['cut_fraction'] for r in results]):.4f}"
          f"  (epsilon={epsilon})")
    print(f"\n  ratio of weights by lag (no cut / Watkins cut):")
    print(f"  {'lag':>5} {'median':>9}  {'[p25, p75]':>20}  {'cut':>9}  {'destroyed':>9}  {'n+1':>6}")
    for lag in LAGS:
        values = np.array([r["ratios"][lag] for r in results if lag in r["ratios"]])
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        low, high = np.percentile(values, [25, 75])
        cut = np.array([r["cut_ratios"][lag] for r in results if lag in r["cut_ratios"]])
        cut = cut[np.isfinite(cut)]
        cut_median = np.median(cut) if cut.size else float("nan")
        gone = np.mean([r["destroyed"][lag] for r in results if lag in r["destroyed"]])
        print(
            f"  {lag:>5} {np.median(values):>9.2f}  [{low:8.2f}, {high:8.2f}]  "
            f"{cut_median:>9.2f}  {gone:>9.2f}  {lag + 1:>6}"
        )

    print(f"\n  fraction of rollouts where Watkins cut destroys TBPTT credit:")
    header = "  " + f"{'epsilon':>8}" + f"{'cut/step':>10}" + "".join(f"{'n=' + str(l):>8}" for l in LAGS)
    print(header)
    for value in EPSILONS:
        frac = np.median([r["sweep"][value]["fraction"] for r in results])
        cells = "".join(
            f"{np.mean([r['sweep'][value]['destroyed'][l] for r in results]):>8.2f}"
            for l in LAGS
        )
        print(f"  {value:>8} {frac:>9.3f}{cells}")

    print(f"\n  coherence |sum term| / sum |term| by lag:")
    print(f"  {'lag':>5} {'median':>9}  {'[p25, p75]':>20}")
    for lag in LAGS:
        values = np.array([r["coherence"][lag] for r in results if lag in r["coherence"]])
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        low, high = np.percentile(values, [25, 75])
        print(f"  {lag:>5} {np.median(values):>9.3f}  [{low:8.3f}, {high:8.3f}]")


if __name__ == "__main__":
    main()
