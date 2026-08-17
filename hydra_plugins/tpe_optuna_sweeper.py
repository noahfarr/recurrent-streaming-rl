from typing import Any, Optional

from hydra.utils import instantiate
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from omegaconf import DictConfig


class TPEOptunaSweeper(OptunaSweeper):
    def __init__(
        self,
        sampler: Any,
        direction: Any,
        storage: Optional[Any],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        custom_search_space: Optional[str],
        params: Optional[DictConfig],
    ) -> None:
        from hydra_plugins.hydra_optuna_sweeper._impl import OptunaSweeperImpl

        self.sweeper = OptunaSweeperImpl(
            instantiate(sampler),
            direction,
            storage,
            study_name,
            n_trials,
            n_jobs,
            max_failure_rate,
            custom_search_space,
            params,
        )
