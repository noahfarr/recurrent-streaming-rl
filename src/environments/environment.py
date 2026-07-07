from hydra.utils import instantiate
from streax.environments import environment
from streax.environments.wrappers import RecordEpisodeStatistics

from src.environments import popgymnax
from src.environments.wrappers import MaskObservationWrapper


def make(namespace, env_id, **kwargs):
    env_kwargs = dict(kwargs.get("kwargs") or {})
    mode = env_kwargs.pop("mode", None)
    if namespace == "popgymnax":
        env, env_params = popgymnax.make(env_id, **env_kwargs)
    else:
        env, env_params = environment.make(f"{namespace}::{env_id}", **env_kwargs)

    if mode in ("P", "V"):
        env = MaskObservationWrapper(env, mode)

    if env_params is not None:
        env_params = env_params.replace(**(kwargs.get("env_params") or {}))

    env = RecordEpisodeStatistics(env)
    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
