import random
import os
from typing import Dict, List
from time import time
from domain.datasets import DatasetEnum
from tqdm.auto import tqdm

from clit_recommender.config import Config
from clit_recommender.data.dataset import EVAL_SIZE, ClitResultDataset, DataRow
from clit_recommender.models.clit_mock import Graph
from clit_recommender.domain.metrics import Metrics


class SingleSystem:
    _config: Config
    _data: List[DataRow]
    _index_map: Dict[str, int]
    _save: bool

    def __init__(self, datasets: List[DatasetEnum], save: bool = True) -> None:
        self._config = Config(depth=1, datasets=datasets)

        data_loader = ClitResultDataset(self._config)
        self._index_map = data_loader._index_map
        data_list = list(data_loader)
        random.seed(self._config.seed)
        random.shuffle(data_list)

        eval_size = int(len(data_list) * self._config.eval_factor)
        self._save = save

        self._data = data_list[:eval_size]

    def run_all(self):
        path = os.path.join(self._config.results_dir, "single_system", str(int(time())))
        for system in self._index_map.keys():
            print("Starting System", system)
            self.run(path, system)

    def run(self, path: str, system: str):
        config = self._config
        config.experiment_name = f"{system.split('/')[-1]}-{int(time())}"

        if self._save:
            path = os.path.join(path, config.experiment_name)

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
            metrics += Metrics.evaluate_results(set(row.actual), result, soft=True)

        print("System", system)
        print(metrics.get_summary())

        if self._save:
            with open(os.path.join(path, "summary.txt"), "w") as f:
                f.write(metrics.get_summary())

            with open(os.path.join(path, "metrics.json"), "w") as f:
                f.write(metrics.to_json())


if __name__ == "__main__":
    single_system = SingleSystem(list(DatasetEnum), save=False)
    single_system.run_all()
