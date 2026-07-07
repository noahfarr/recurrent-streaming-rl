from hydra.core.hydra_config import HydraConfig

from src.recipes import (
    ac_lambda_brax,
    ac_lambda_bsuite,
    ac_lambda_popgymnax,
    ppo_brax,
    ppo_popgymnax,
    q_lambda_popgymnax,
    qrc_lambda_bsuite,
    qrc_lambda_popgymnax,
    sac_brax,
    fast_sac_brax,
)

register = {
    ("ppo", "popgymnax"): ppo_popgymnax.make,
    ("ppo", "brax"): ppo_brax.make,
    ("sac", "brax"): sac_brax.make,
    ("fast_sac", "brax"): fast_sac_brax.make,
    ("qrc_lambda", "bsuite"): qrc_lambda_bsuite.make,
    ("qrc_lambda", "popgymnax"): qrc_lambda_popgymnax.make,
    ("q_lambda", "popgymnax"): q_lambda_popgymnax.make,
    ("ac_lambda", "bsuite"): ac_lambda_bsuite.make,
    ("ac_lambda", "popgymnax"): ac_lambda_popgymnax.make,
    ("ac_lambda", "brax"): ac_lambda_brax.make,
}


def make(cfg):
    suite = cfg.environment.get("suite", cfg.environment.namespace)
    name = HydraConfig.get().runtime.choices["algorithm"]
    key = (name, suite)
    return register[key](cfg)
