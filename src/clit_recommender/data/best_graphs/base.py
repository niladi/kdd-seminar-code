from typing import List, Optional, Self
from os.path import join

from clit_recommender import BEST_GRAPHS_PATH
from clit_recommender.config import Config
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.systems import System
from clit_recommender.util import create_hot_vector


class BestGraphBase:
    datasets: List[Dataset]
    systems: List[System]
    path: Optional[str] = None

    def __init__(self, datasets: List[Dataset], systems: List[Dataset]) -> None:
        self.datasets = datasets
        self.systems = systems
        if self.path is None:
            self.path = join(
                BEST_GRAPHS_PATH, create_hot_vector(self.systems, len(list(System)))
            )

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(config.datasets, config.systems)
