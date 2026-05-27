from omegaconf import open_dict

from src.algorithms.ppo import ppo_brax, ppo_popjym
from src.algorithms.qrc import qrc_bsuite, qrc_popjym
from src.algorithms.stream_ac import (
    stream_ac_brax,
    stream_ac_bsuite,
    stream_ac_popjym,
)

register = {
    ("ppo", "popjym"): ppo_popjym.make,
    ("ppo", "brax"): ppo_brax.make,
    ("qrc", "bsuite"): qrc_bsuite.make,
    ("qrc", "popjym"): qrc_popjym.make,
    ("stream_ac", "bsuite"): stream_ac_bsuite.make,
    ("stream_ac", "popjym"): stream_ac_popjym.make,
    ("stream_ac", "brax"): stream_ac_brax.make,
}


def make(cfg):
    family = cfg.environment.get("suite", cfg.environment.namespace)
    name = cfg.algorithm.name
    with open_dict(cfg):
        del cfg.algorithm.name
    key = (name, family)
    return register[key](cfg)
