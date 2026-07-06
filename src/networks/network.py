from typing import Callable

import flax.linen as nn
from hydra.utils import instantiate

from src.utils.identity import identity
from src.utils.typing import Carry, Key


class Network(nn.Module):
    feature_extractor: Callable = identity
    cell: nn.Module | None = None
    head: Callable = identity

    @nn.compact
    def __call__(self, carry, obs, action, reward, done):
        x = self.feature_extractor(obs, action, reward, done)
        if self.cell is None:
            return carry, self.head(x)
        carry, x = self.cell(carry, x, done=done)
        return carry, self.head(x)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple = ()) -> Carry:
        if self.cell is None:
            return None
        return self.cell.initialize_carry(key, input_shape)


class ObservationFeatureExtractor(nn.Module):
    """Adapts an obs-only module to Network's (obs, action, reward, done) call."""

    layers: Callable

    @nn.compact
    def __call__(self, obs, action, reward, done):
        return self.layers(obs)


def build_cell(cfg):
    """Builds the (possibly RTRL/RNN-wrapped) recurrent cell from cfg.cell/cfg.mode."""
    if cfg.cell.name == "ffn":
        return None
    raw_cell = instantiate(cfg.cell)
    return instantiate(cfg.mode.wrapper)(cell=raw_cell)
