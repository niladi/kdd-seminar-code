import random
import os
from typing import Dict, List
from time import time
from tqdm.auto import tqdm

from clit_recommender.config import Config
from clit_recommender.data.dataset import EVAL_SIZE, ClitRecommenderDataset, DataRow
from clit_recommender.models.clit_mock import Graph
from clit_recommender.domain.metrics import Metrics


class SingleSystem:
    _config: Config
    _data: List[DataRow]
    _index_map: Dict[str, int]

    def __init__(self) -> None:
        self._config = Config(depth=1)

        data_loader = ClitRecommenderDataset(self._config)
        self._index_map = data_loader._index_map
        data_list = list(data_loader)
        random.seed(self._config.seed)
        random.shuffle(data_list)

        self._data = data_list[:EVAL_SIZE]

    def run_all(self):
        for system in self._index_map.keys():
            print("Starting System", system)
            self.run(system)

    def run(self, system: str):
        config = self._config
        config.experiment_name = f"{system.split('/')[-1]}-{int(time())}"

        path = os.path.join(config.cache_dir, "single_system", config.experiment_name)

        os.makedirs(path)

        with open(os.path.join(path, "config.json"), "w") as f:
            f.write(config.to_json())

        last_level = [0.0] * len(self._index_map)
        last_level[self._index_map[system]] = 1.0

        g = Graph.create_1_dim(config, [], last_level, 0)

        row: DataRow
        metrics = Metrics.zeros()
        for (row,) in tqdm(self._data):
            result = g.forward(row)
            metrics += Metrics.evaluate_results(set(row.actual), result)

        print("System", system)
        print(metrics.get_summary())

        with open(os.path.join(path, "summary.txt"), "w") as f:
            f.write(metrics.get_summary())

        with open(os.path.join(path, "metrics.json"), "w") as f:
            f.write(metrics.to_json())


if __name__ == "__main__":
    single_system = SingleSystem()
    for system in single_system._index_map.keys():
        single_system.run(system)
