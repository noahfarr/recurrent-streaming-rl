from typing import Callable

import flax.linen as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict

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


def build_cell(cfg, input_size=None):
    if HydraConfig.get().runtime.choices["cell"] == "ffn":
        return None
    if input_size is not None:
        with open_dict(cfg):
            cfg.cell.config.features = input_size
    raw_cell = instantiate(cfg.cell)
    return instantiate(cfg.mode.wrapper)(cell=raw_cell)
