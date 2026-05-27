from hydra.utils import instantiate
from memorax.environments import RecordEpisodeStatistics, environment

from src.environments.k_memory_chain import (
    KMemoryChain,
    EnvParams as KMemoryChainEnvParams,
)


registry = {
    "KMemoryChain-v0": (KMemoryChain, KMemoryChainEnvParams),
}


def make(namespace, env_id, **kwargs):
    if namespace == "rsrl" and env_id in registry:
        env_cls, params_cls = registry[env_id]
        env = env_cls(**(kwargs.get("kwargs") or {}))
        env_params = env.default_params.replace(**(kwargs.get("env_params") or {}))
    else:
        env_id = f"{namespace}::{env_id}"
        env, env_params = environment.make(env_id, **(kwargs.get("kwargs") or {}))

        if env_params is not None:
            env_params = env_params.replace(**(kwargs.get("env_params") or {}))

    env = RecordEpisodeStatistics(env)
    for wrapper in kwargs.get("wrappers", []):
        env = instantiate(wrapper, env)

    return env, env_params
