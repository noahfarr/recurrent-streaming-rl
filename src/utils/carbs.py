from __future__ import annotations

import glob
import json
import logging
import math
from typing import List, Tuple

import attr
import numpy as np
import torch
from carbs import (
    CARBS,
    SUGGESTION_ID_DICT_KEY,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    Param,
    RealNumberSpace,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

EPSILON = 1e-6


@attr.s(auto_attribs=True, hash=True)
class Pow2Space(RealNumberSpace):
    min: float = 1.0
    max: float = float("+inf")

    def basic_from_param(self, value) -> float:
        return math.log2(value) / self.scale

    def param_from_basic(self, value: float, is_rounded: bool = True):
        exponent = value * self.scale
        if is_rounded:
            return int(2 ** round(exponent))
        return 2**exponent

    def round_tensor_in_basic(self, value: torch.Tensor) -> torch.Tensor:
        return torch.round(value * self.scale) / self.scale

    @property
    def plot_scale(self) -> str:
        return "log"


@attr.s(auto_attribs=True, hash=True)
class CategoricalSpace(RealNumberSpace):
    choices: Tuple = ()

    def _index(self, value):
        if value in self.choices:
            return self.choices.index(value)
        return float(value)

    def basic_from_param(self, value) -> float:
        return self._index(value) / self.scale

    def param_from_basic(self, value: float, is_rounded: bool = True):
        idx = int(round(value * self.scale))
        idx = max(0, min(len(self.choices) - 1, idx))
        return self.choices[idx]

    def round_tensor_in_basic(self, value: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(value * self.scale)
        rounded = torch.clamp(rounded, 0, max(len(self.choices) - 1, 0))
        return rounded / self.scale

    @property
    def plot_scale(self) -> str:
        return "linear"


SPACES = {
    "uniform": (LinearSpace, False),
    "int_uniform": (LinearSpace, True),
    "uniform_pow2": (Pow2Space, True),
    "log_normal": (LogSpace, False),
    "logit_normal": (LogitSpace, False),
}


def make_space(pc) -> RealNumberSpace:
    distribution = str(pc.distribution)
    if distribution == "categorical":
        choices = tuple(pc.choices)
        return CategoricalSpace(
            min=0,
            max=max(len(choices) - 1, 0),
            scale=1.0,
            is_integer=True,
            choices=choices,
        )
    if distribution not in SPACES:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    space_class, is_integer = SPACES[distribution]
    raw_scale = pc.get("scale", "auto")
    if raw_scale in ("auto", "time"):
        if space_class is LinearSpace:
            scale = pc.max - pc.min
        elif space_class is Pow2Space:
            scale = math.log2(pc.max) - math.log2(pc.min)  # octaves -> basic span ~1, like LinearSpace
        else:
            scale = 1.0
    else:
        scale = raw_scale
    return space_class(min=pc.min, max=pc.max, scale=scale, is_integer=is_integer)


def search_center(pc, space, default_value):
    if isinstance(space, CategoricalSpace):
        chosen = pc.get("search_center", default_value)
        if chosen in space.choices:
            return chosen
        return space.choices[len(space.choices) // 2]

    explicit = pc.get("search_center", None)
    if explicit is not None:
        value = float(explicit)
    elif default_value is not None:
        value = min(max(float(default_value), space.min), space.max)
    else:
        value = space.param_from_basic(
            (space.basic_from_param(space.min) + space.basic_from_param(space.max)) / 2
        )
    return space.param_from_basic(space.basic_from_param(value))


def make_param(name: str, pc, default_value=None) -> Param:
    space = make_space(pc)
    return Param(
        name=name, space=space, search_center=search_center(pc, space, default_value)
    )


class CarbsOptimizer:
    def __init__(self, carbs: CARBS):
        self.carbs = carbs

    def suggest(self) -> Tuple[str, dict]:
        suggestion = dict(self.carbs.suggest().suggestion)
        suggestion_id = suggestion.pop(SUGGESTION_ID_DICT_KEY)
        return suggestion_id, suggestion

    def observe(
        self, suggestion_id, hyperparameters, score, cost, is_failure: bool = False
    ) -> None:
        self.carbs.observe(
            ObservationInParam(
                input={**hyperparameters, SUGGESTION_ID_DICT_KEY: suggestion_id},
                output=float(score),
                cost=float(cost),
                is_failure=is_failure,
            )
        )


def load_observations(patterns, defaults: dict) -> List[ObservationInParam]:
    if isinstance(patterns, str):
        patterns = [patterns]

    param_names = list(defaults)
    files = sorted({path for pattern in patterns for path in glob.glob(str(pattern))})
    observations: List[ObservationInParam] = []
    skipped = 0
    backfilled = 0
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                hyperparameters = record.get("hyperparameters")
                if not isinstance(hyperparameters, dict):
                    skipped += 1
                    continue
                input_values = {}
                for name in param_names:
                    if name in hyperparameters:
                        input_values[name] = hyperparameters[name]
                    else:
                        input_values[name] = defaults[name]
                        backfilled += 1
                observations.append(
                    ObservationInParam(
                        input=input_values,
                        output=float(record.get("score") or 0.0),
                        cost=float(record.get("cost") or 0.0),
                        is_failure=bool(record.get("is_failure", False)),
                    )
                )

    logger.info(
        "warm start: loaded %d observations from %d file(s) "
        "(%d skipped, %d fields backfilled to search center)",
        len(observations),
        len(files),
        skipped,
        backfilled,
    )
    return observations


def build_carbs(cfg):
    sweep = cfg.sweep
    params = [
        make_param(name, pc, OmegaConf.select(cfg, name, default=None))
        for name, pc in sweep.params.items()
    ]

    config = instantiate(sweep.carbs)
    carbs = CARBS(config, params)

    if sweep.prior_score is not None and sweep.prior_cost is not None:
        prior_input = {param.name: param.search_center for param in params}
        carbs.initialize_from_observations(
            [
                ObservationInParam(
                    input=prior_input,
                    output=float(sweep.prior_score),
                    cost=float(sweep.prior_cost),
                )
            ]
        )

    warm_start = sweep.get("warm_start", None)
    if warm_start:
        if OmegaConf.is_config(warm_start):
            warm_start = OmegaConf.to_container(warm_start, resolve=True)
        defaults = {param.name: param.search_center for param in params}
        observations = load_observations(warm_start, defaults)
        if observations:
            carbs.initialize_from_observations(observations)

    return CarbsOptimizer(carbs), sweep.metric, sweep.cost


def basic_bounds(space: RealNumberSpace) -> Tuple[float, float]:
    return space.basic_from_param(space.min_bound), space.basic_from_param(
        space.max_bound
    )


def to_unit(space: RealNumberSpace, value) -> float:
    lo, hi = basic_bounds(space)
    basic = space.basic_from_param(value)
    return float(np.clip((basic - lo) / (hi - lo + EPSILON), 0.0, 1.0))


def from_unit(space: RealNumberSpace, n: float):
    lo, hi = basic_bounds(space)
    return space.param_from_basic(lo + n * (hi - lo))


def pareto_points(observations):
    if not observations:
        return [], []
    scores = np.array([e["output"] for e in observations])
    costs = np.array([e["cost"] for e in observations])
    pareto, idxs = [], []
    best_score = -np.inf
    for i in np.argsort(costs):
        if scores[i] > best_score + EPSILON:
            pareto.append(observations[i])
            idxs.append(int(i))
            best_score = scores[i]
    return pareto, idxs
