from omegaconf import open_dict

from src.recipes import (
    ppo_brax,
    ppo_popgymnax,
    q_lambda_popgymnax,
    qrc_bsuite,
    qrc_popgymnax,
    stream_ac_brax,
    stream_ac_bsuite,
    stream_ac_popgymnax,
)

register = {
    ("ppo", "popgymnax"): ppo_popgymnax.make,
    ("ppo", "brax"): ppo_brax.make,
    ("qrc_lambda", "bsuite"): qrc_bsuite.make,
    ("qrc_lambda", "popgymnax"): qrc_popgymnax.make,
    ("q_lambda", "popgymnax"): q_lambda_popgymnax.make,
    ("ac_lambda", "bsuite"): stream_ac_bsuite.make,
    ("ac_lambda", "popgymnax"): stream_ac_popgymnax.make,
    ("ac_lambda", "brax"): stream_ac_brax.make,
}


def make(cfg):
    family = cfg.environment.get("suite", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg)
