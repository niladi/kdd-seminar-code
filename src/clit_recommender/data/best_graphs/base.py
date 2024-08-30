from dataclasses import dataclass
from operator import attrgetter
from typing import List, Optional, Self
from os.path import join

from dataclasses_json import dataclass_json

from clit_recommender import BEST_GRAPHS_PATH
from clit_recommender.config import Config
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.systems import System
from clit_recommender.util import create_hot_vector


class BestGraphBase:
    config: Config
    path: Optional[str] = None

    def __init__(self, config: Config) -> None:
        self.config = config
        if self.path is None:
            self.path = join(
                BEST_GRAPHS_PATH,
                create_hot_vector(
                    list(map(attrgetter("index"), self.config.systems)),
                    len(list(System)),
                ),
            )
