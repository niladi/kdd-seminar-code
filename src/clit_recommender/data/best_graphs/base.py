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
        if config.fixed_size_combo > 0:
            self.path = join(self.path, f"fixed_size_combo_{config.fixed_size_combo}")
        #if config.margin_range > 0:
        #    self.path = join(self.path, f"margin_range_{config.margin_range")
