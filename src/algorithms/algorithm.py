from hydra.core.hydra_config import HydraConfig

from src.recipes import (
    ac_lambda_brax,
    ac_lambda_bsuite,
    ac_lambda_popgymnax,
    fast_sac_brax,
    ppo_brax,
    ppo_popgymnax,
    q_lambda_popgymnax,
    qrc_lambda_bsuite,
    qrc_lambda_popgymnax,
    sac_brax,
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


def policy_params(state):
    name = HydraConfig.get().runtime.choices["algorithm"]
    return getattr(
        state,
        {
            "ppo": "actor_params",
            "sac": "actor_params",
            "fast_sac": "actor_params",
            "ac_lambda": "actor_params",
            "q_lambda": "q_params",
            "qrc_lambda": "q_params",
        }[name],
    )
