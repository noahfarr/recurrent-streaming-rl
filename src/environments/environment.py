from hydra.utils import instantiate
from streax.environments import environment
from streax.environments.wrappers import RecordEpisodeStatistics

from src.environments import popgymnax


def make(namespace, env_id, **kwargs):
    env_kwargs = kwargs.get("kwargs") or {}
    if namespace == "popgymnax":
        env, env_params = popgymnax.make(env_id, **env_kwargs)
    else:
        env, env_params = environment.make(f"{namespace}::{env_id}", **env_kwargs)

    if env_params is not None:
        env_params = env_params.replace(**(kwargs.get("env_params") or {}))

    env = RecordEpisodeStatistics(env)
    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
