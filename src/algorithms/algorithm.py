from omegaconf import open_dict

from src.recipes import (
    ppo_brax,
    ppo_popgymnax,
    qrc_bsuite,
    qrc_popgymnax,
    stream_ac_brax,
    stream_ac_bsuite,
    stream_ac_popgymnax,
)

register = {
    ("ppo", "popgymnax"): ppo_popgymnax.make,
    ("ppo", "brax"): ppo_brax.make,
    ("qrc", "bsuite"): qrc_bsuite.make,
    ("qrc", "popgymnax"): qrc_popgymnax.make,
    ("stream_ac", "bsuite"): stream_ac_bsuite.make,
    ("stream_ac", "popgymnax"): stream_ac_popgymnax.make,
    ("stream_ac", "brax"): stream_ac_brax.make,
}


def make(cfg):
    family = cfg.environment.get("suite", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg)
