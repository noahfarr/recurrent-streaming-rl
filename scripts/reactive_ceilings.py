import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from popgymnax.environments import (
    popgym_autoencode,
    popgym_battleship,
    popgym_concentration,
    popgym_count_recall,
    popgym_higherlower,
    popgym_minesweeper,
    popgym_multiarmedbandit,
    popgym_repeat_first,
)

EPISODES = 1024
HELDOUT = 16384
RESTARTS = 8


def reward_sign(reward, first):
    return jnp.where(first, 2, jnp.where(reward > 0, 1, 0))


def obs_index_repeat_first(obs):
    return jnp.argmax(obs)


def obs_index_autoencode(obs):
    return jnp.where(obs.sum() == 0, 4, jnp.argmax(obs))


def obs_index_higher_lower(obs):
    return jnp.argmax(obs)


def obs_index_count_recall(obs):
    return jnp.argmax(obs[:2]) * 2 + jnp.argmax(obs[2:4])


def obs_index_battleship(obs):
    return (obs[0] > 0).astype(jnp.int32)


def obs_index_minesweeper(obs):
    return jnp.argmax(obs)


def obs_index_bandit(obs):
    return (obs[0] > 0).astype(jnp.int32)


def cycle_init(num_obs, num_actions, stride):
    table = np.zeros((num_obs * num_actions * 2,), dtype=np.int32)
    for obs_index in range(num_obs):
        for prev_action in range(num_actions):
            for first in range(2):
                table[obs_index * num_actions * 2 + prev_action * 2 + first] = (
                    prev_action + stride
                ) % num_actions
    return jnp.asarray(table)


def stay_shift_init(num_obs, num_actions, stride):
    table = np.zeros((num_obs * num_actions * 2,), dtype=np.int32)
    for obs_index in range(num_obs):
        for prev_action in range(num_actions):
            for first in range(2):
                action = (
                    prev_action
                    if obs_index == 1
                    else (prev_action + stride) % num_actions
                )
                table[obs_index * num_actions * 2 + prev_action * 2 + first] = action
    return jnp.asarray(table)


def action_ceiling(name, env, obs_index_fn, num_obs, horizon, extra_inits, sweeps=8):
    params = env.default_params
    num_actions = env.action_space(params).n
    _, action_fn, _ = make_contexts(obs_index_fn, num_actions)
    num_contexts = num_obs * num_actions * 2
    inits = list(extra_inits)
    inits += [
        jax.random.randint(k, (num_contexts,), 0, num_actions, dtype=jnp.int32)
        for k in jax.random.split(jax.random.key(17), 2)
    ]
    heldout = jax.jit(
        functools.partial(
            evaluate,
            env,
            params,
            context_fn=action_fn,
            horizon=horizon,
            keys=jax.random.split(jax.random.key(1000), HELDOUT),
        )
    )
    value, _ = ascend(
        env, params, action_fn, num_contexts, num_actions, horizon, inits, sweeps=sweeps
    )
    for index, init in enumerate(extra_inits):
        baseline = float(heldout(init))
        print(f"  init {index}: {baseline:+.4f}", flush=True)
        value = max(value, baseline)
    print(f"{name:<14} {value:>+14.3f}  (pi(a|o,a,t0))")
    return value


def battleship_ceiling():
    env = popgym_battleship.BattleshipEasy()
    inits = [cycle_init(2, 64, stride) for stride in (1, 3, 5, 8)]
    return action_ceiling("Battleship", env, obs_index_battleship, 2, 64, inits)


def minesweeper_ceiling():
    env = popgym_minesweeper.MineSweeperEasy()
    inits = [cycle_init(3, 16, stride) for stride in (1, 3, 5, 7)]
    return action_ceiling("MineSweeper", env, obs_index_minesweeper, 3, 16, inits)


def bandit_ceiling():
    env = popgym_multiarmedbandit.MultiarmedBanditEasy()
    inits = [cycle_init(2, 10, 1)]
    inits += [stay_shift_init(2, 10, stride) for stride in (1, 3)]
    return action_ceiling("MultiArmedBandit", env, obs_index_bandit, 2, 200, inits)


def make_contexts(obs_index_fn, num_actions):
    def obs_only(obs, prev_action, prev_reward, first):
        return obs_index_fn(obs)

    def with_action(obs, prev_action, prev_reward, first):
        return obs_index_fn(obs) * (num_actions * 2) + prev_action * 2 + first.astype(jnp.int32)

    def with_reward(obs, prev_action, prev_reward, first):
        return (
            obs_index_fn(obs) * (num_actions * 3)
            + prev_action * 3
            + reward_sign(prev_reward, first)
        )

    return obs_only, with_action, with_reward


def evaluate(env, params, table, context_fn, horizon, keys):
    def episode(key):
        obs, state = env.reset_env(key, params)

        def body(carry, _):
            obs, state, prev_action, prev_reward, first, done, total = carry
            action = table[context_fn(obs, prev_action, prev_reward, first)]
            next_obs, next_state, reward, terminated, _ = env.step_env(
                key, state, action, params
            )
            total = total + jnp.where(done, 0.0, reward)
            return (
                next_obs,
                next_state,
                action,
                reward,
                jnp.array(False),
                jnp.logical_or(done, terminated),
                total,
            ), None

        carry = (obs, state, 0, 0.0, jnp.array(True), jnp.array(False), 0.0)
        (_, _, _, _, _, _, total), _ = jax.lax.scan(body, carry, None, length=horizon)
        return total

    return jnp.mean(jax.vmap(episode)(keys))


def ascend(env, params, context_fn, num_contexts, num_actions, horizon, inits, sweeps=20):
    train = jax.jit(
        functools.partial(
            evaluate,
            env,
            params,
            context_fn=context_fn,
            horizon=horizon,
            keys=jax.random.split(jax.random.key(0), EPISODES),
        )
    )
    heldout = jax.jit(
        functools.partial(
            evaluate,
            env,
            params,
            context_fn=context_fn,
            horizon=horizon,
            keys=jax.random.split(jax.random.key(1000), HELDOUT),
        )
    )
    batched = jax.jit(jax.vmap(train))

    best_table, best_train = None, -np.inf
    for init in inits:
        table = jnp.asarray(init, dtype=jnp.int32)
        value = float(train(table))
        for _ in range(sweeps):
            improved = False
            for context in range(num_contexts):
                candidates = jnp.stack(
                    [table.at[context].set(action) for action in range(num_actions)]
                )
                scores = np.asarray(batched(candidates))
                action = int(scores.argmax())
                if scores[action] > value + 1e-9:
                    value, table, improved = float(scores[action]), candidates[action], True
            if not improved:
                break
        if value > best_train:
            best_table, best_train = table, value
    return float(heldout(best_table)), best_table


def restart_inits(key, num_contexts, num_actions, extra):
    keys = jax.random.split(key, RESTARTS)
    inits = [jnp.zeros((num_contexts,), dtype=jnp.int32)]
    inits += [
        jax.random.randint(k, (num_contexts,), 0, num_actions, dtype=jnp.int32) for k in keys
    ]
    return list(extra) + inits


def lift(coarse, num_obs_contexts, num_actions, source_slots, target_slots, slot_map):
    fine = np.zeros((num_obs_contexts * num_actions * target_slots,), dtype=np.int32)
    coarse = np.asarray(coarse)
    for obs_index in range(num_obs_contexts):
        for prev_action in range(num_actions):
            for slot in range(target_slots):
                if source_slots == 1:
                    value = coarse[obs_index]
                else:
                    value = coarse[
                        obs_index * num_actions * source_slots
                        + prev_action * source_slots
                        + slot_map(slot)
                    ]
                fine[
                    obs_index * num_actions * target_slots + prev_action * target_slots + slot
                ] = value
    return jnp.asarray(fine)


def concentration_random(env, params, keys):
    def episode(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, params)

        def body(carry, step_key):
            obs, state, done, total = carry
            hidden = jnp.reshape(obs, (env.num_cards, env.num_types + 1))[:, env.num_types]
            action = jax.random.choice(
                step_key, env.num_cards, p=hidden / jnp.maximum(hidden.sum(), 1.0)
            )
            next_obs, next_state, reward, terminated, _ = env.step_env(
                step_key, state, action, params
            )
            total = total + jnp.where(done, 0.0, reward)
            return (next_obs, next_state, jnp.logical_or(done, terminated), total), None

        step_keys = jax.random.split(key, env.episode_length)
        (_, _, _, total), _ = jax.lax.scan(body, (obs, state, jnp.array(False), 0.0), step_keys)
        return total

    return float(jnp.mean(jax.vmap(episode)(keys)))


def concentration_deterministic(env, params, keys):
    def episode(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, params)

        def body(carry, step_key):
            obs, state, done, total = carry
            hidden = jnp.reshape(obs, (env.num_cards, env.num_types + 1))[:, env.num_types]
            action = jnp.argmax(hidden)
            next_obs, next_state, reward, terminated, _ = env.step_env(
                step_key, state, action, params
            )
            total = total + jnp.where(done, 0.0, reward)
            return (next_obs, next_state, jnp.logical_or(done, terminated), total), None

        step_keys = jax.random.split(key, env.episode_length)
        (_, _, _, total), _ = jax.lax.scan(body, (obs, state, jnp.array(False), 0.0), step_keys)
        return total

    return float(jnp.mean(jax.vmap(episode)(keys)))


def concentration_ceiling():
    env = popgym_concentration.ConcentrationEasy()
    params = env.default_params
    keys = jax.random.split(jax.random.key(1000), HELDOUT)
    uniform = concentration_random(env, params, keys)
    lowest = concentration_deterministic(env, params, keys)
    print(f"  uniform over hidden:       {uniform:+.4f}")
    print(f"  lowest-index hidden:       {lowest:+.4f}")
    value = max(uniform, lowest)
    print(f"{'Concentration':<14} {value:>+14.3f}  (reactive, exchangeable)")
    return value


def main():
    if len(sys.argv) > 1:
        selected = {
            "battleship": battleship_ceiling,
            "minesweeper": minesweeper_ceiling,
            "bandit": bandit_ceiling,
            "concentration": concentration_ceiling,
        }
        for name in sys.argv[1].split(","):
            selected[name]()
        return
    tasks = [
        ("Autoencode", popgym_autoencode.AutoencodeEasy(), obs_index_autoencode, 5, 4, 103),
        ("CountRecall", popgym_count_recall.CountRecallEasy(), obs_index_count_recall, 4, 27, 51),
        ("HigherLower", popgym_higherlower.HigherLowerEasy(), obs_index_higher_lower, 13, 2, 51),
        ("RepeatFirst", popgym_repeat_first.RepeatFirstEasy(), obs_index_repeat_first, 4, 4, 51),
    ]

    print(f"{'task':<14} {'pi(a|o)':>10} {'pi(a|o,a,t0)':>14} {'pi(a|o,a,r)':>13}")
    for name, env, obs_index_fn, num_obs_contexts, num_actions, horizon in tasks:
        params = env.default_params
        obs_only_fn, action_fn, reward_fn = make_contexts(obs_index_fn, num_actions)
        key = jax.random.key(hash(name) % (2**31))

        value_obs, table_obs = ascend(
            env,
            params,
            obs_only_fn,
            num_obs_contexts,
            num_actions,
            horizon,
            restart_inits(key, num_obs_contexts, num_actions, []),
        )
        lifted = lift(table_obs, num_obs_contexts, num_actions, 1, 2, lambda slot: slot)
        value_action, table_action = ascend(
            env,
            params,
            action_fn,
            num_obs_contexts * num_actions * 2,
            num_actions,
            horizon,
            restart_inits(key, num_obs_contexts * num_actions * 2, num_actions, [lifted]),
        )
        lifted = lift(
            table_action,
            num_obs_contexts,
            num_actions,
            2,
            3,
            lambda slot: 1 if slot == 2 else 0,
        )
        value_reward, _ = ascend(
            env,
            params,
            reward_fn,
            num_obs_contexts * num_actions * 3,
            num_actions,
            horizon,
            restart_inits(key, num_obs_contexts * num_actions * 3, num_actions, [lifted]),
        )
        print(f"{name:<14} {value_obs:>+10.3f} {value_action:>+14.3f} {value_reward:>+13.3f}")

    environment = popgym_concentration.ConcentrationEasy()
    value = concentration_random(
        environment,
        environment.default_params,
        jax.random.split(jax.random.key(1000), HELDOUT),
    )
    print(f"{'Concentration':<14} {'':>10} {value:>+14.3f}  (uniform-random reactive)")


if __name__ == "__main__":
    main()
