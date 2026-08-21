from hydra.utils import instantiate
from streamlet.environments import environment
from streamlet.environments.wrappers import LogAverageReward, RecordEpisodeStatistics

from src.environments import ale, brax, mujoco, popgymnax

registry = {
    "popgymnax": popgymnax.make,
    "brax": brax.make,
    "ale": ale.make,
    "gymnasium": mujoco.make,
}


def make(namespace, env_id, **kwargs):
    env_kwargs = dict(kwargs.get("kwargs") or {})
    env, env_params = registry.get(
        namespace,
        lambda env_id, **kw: environment.make(f"{namespace}::{env_id}", **kw),
    )(env_id, **env_kwargs)

    if env_params is not None:
        env_params = env_params.replace(**(kwargs.get("env_params") or {}))

    env = RecordEpisodeStatistics(env)
    env = LogAverageReward(env)
    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
