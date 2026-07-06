from omegaconf import open_dict

from src.recipes import (
    ppo_brax,
    ppo_popjym,
)

register = {
    ("ppo", "popjym"): ppo_popjym.make,
    ("ppo", "brax"): ppo_brax.make,
}


def make(cfg):
    family = cfg.environment.get("suite", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg)
