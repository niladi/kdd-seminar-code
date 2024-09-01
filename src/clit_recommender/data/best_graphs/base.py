from operator import attrgetter
from typing import Optional
from os.path import join


from clit_recommender import BEST_GRAPHS_PATH
from clit_recommender.config import Config

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
                    config.md_modules_count,
                ),
                config.metric_type.lower(),
            )
