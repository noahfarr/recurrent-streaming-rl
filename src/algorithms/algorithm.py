from hydra.core.hydra_config import HydraConfig

from src.recipes import (
    stream_q_atari,
    smg_mujoco,
    ppo_brax,
    ppo_classic_control,
    ppo_minatar,
    ppo_popgymnax,
    qrc_bsuite,
    qrc_minatar,
    qrc_popgymnax,
    sac_brax,
    stream_ac_brax,
    stream_ac_bsuite,
    stream_ac_minatar,
    stream_ac_mujoco,
    stream_ac_popgymnax,
    stream_q_minatar,
    stream_q_popgymnax,
)

register = {
    ("ppo", "popgymnax"): ppo_popgymnax.make,
    ("ppo", "brax"): ppo_brax.make,
    ("ppo", "minatar"): ppo_minatar.make,
    ("ppo", "classic_control"): ppo_classic_control.make,
    ("sac", "brax"): sac_brax.make,
    ("qrc", "bsuite"): qrc_bsuite.make,
    ("qrc", "popgymnax"): qrc_popgymnax.make,
    ("qrc", "minatar"): qrc_minatar.make,
    ("stream_q", "popgymnax"): stream_q_popgymnax.make,
    ("stream_q", "ale"): stream_q_atari.make,
    ("intentional_q", "ale"): stream_q_atari.make,
    ("implicit_q", "ale"): stream_q_atari.make,
    ("implicit_q", "popgymnax"): stream_q_popgymnax.make,
    ("stream_q", "minatar"): stream_q_minatar.make,
    ("intentional_q", "minatar"): stream_q_minatar.make,
    ("adaptive_q", "minatar"): stream_q_minatar.make,
    ("implicit_q", "minatar"): stream_q_minatar.make,
    ("kalman_q", "minatar"): stream_q_minatar.make,
    ("calibrated_q", "minatar"): stream_q_minatar.make,
    ("stream_ac", "bsuite"): stream_ac_bsuite.make,
    ("intentional_ac", "bsuite"): stream_ac_bsuite.make,
    ("adaptive_ac", "bsuite"): stream_ac_bsuite.make,
    ("stream_ac", "popgymnax"): stream_ac_popgymnax.make,
    ("stream_ac", "brax"): stream_ac_brax.make,
    ("stream_ac", "minatar"): stream_ac_minatar.make,
    ("stream_ac", "foragax"): stream_ac_minatar.make,
    ("intentional_ac", "foragax"): stream_ac_minatar.make,
    ("implicit_ac", "foragax"): stream_ac_minatar.make,
    ("stream_ac", "mujoco"): stream_ac_mujoco.make,
    ("intentional_ac", "brax"): stream_ac_brax.make,
    ("implicit_ac", "brax"): stream_ac_brax.make,
    ("intentional_ac", "minatar"): stream_ac_minatar.make,
    ("intentional_ac", "mujoco"): stream_ac_mujoco.make,
    ("smg_lambda", "mujoco"): smg_mujoco.make,
    ("implicit_ac", "mujoco"): stream_ac_mujoco.make,
    ("adaptive_ac", "minatar"): stream_ac_minatar.make,
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
            "stream_ac": "params",
            "intentional_ac": "params",
            "adaptive_ac": "params",
            "implicit_ac": "params",
            "smg_lambda": "actor_params",
            "stream_q": "params",
            "intentional_q": "params",
            "adaptive_q": "params",
            "implicit_q": "params",
            "calibrated_q": "params",
            "kalman_q": "params",
            "qrc": "params",
        }[name],
    )
