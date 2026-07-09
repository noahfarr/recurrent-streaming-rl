from pathlib import Path

import gymnax
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf


def get_action_dim(cfg):
    from src import environment

    env, env_params = environment.make(**cfg)

    if isinstance(env.action_space(env_params), gymnax.environments.spaces.Discrete):
        action_dim = env.action_space(env_params).n
    else:
        action_dim, *_ = env.action_space(env_params).shape
    return action_dim


def cascading_fallback(group: str, algorithm: str, environment: str, cell=None) -> str:
    gh = GlobalHydra.instance()
    loader = gh.config_loader()

    parts = environment.split("/")
    while parts:
        parent = "/".join(parts[:-1])
        leaf = parts[-1]
        search = f"{group}/{algorithm}/{parent}" if parent else f"{group}/{algorithm}"

        if cell:
            cell_options = loader.get_group_options(f"{search}/{leaf}")
            if cell in cell_options:
                return f"{algorithm}/{'/'.join(parts)}/{cell}"

        if leaf in loader.get_group_options(search):
            return f"{algorithm}/{'/'.join(parts)}"

        parts.pop()

    return f"{algorithm}"


def get_group(_root_):
    algorithm = HydraConfig.get().runtime.choices["algorithm"]
    group = f"{algorithm}_{_root_.environment.namespace}_{_root_.environment.env_id}"

    # if _root_.environment.get("kwargs"):
    #     kwargs = {
    #         "_".join(
    #             f"{k}_{v}"
    #             for k, v in sorted(_root_.environment.get("kwargs", {}).items())
    #         )
    #     }
    #     group += f"_{kwargs}"

    group = group[:128]
    return group


def groups():
    choices = HydraConfig.get().runtime.choices
    return {k: v for k, v in choices.items() if not k.startswith("hydra/")}


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_action_dim", get_action_dim)
OmegaConf.register_new_resolver("cascading_fallback", cascading_fallback)
OmegaConf.register_new_resolver("get_group", get_group)
OmegaConf.register_new_resolver("groups", groups)
