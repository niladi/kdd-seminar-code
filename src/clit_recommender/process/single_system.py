from operator import attrgetter
import random
import os
from typing import Dict, List
from time import time

from tqdm.auto import tqdm

from clit_recommender.data.dataset.clit_result_dataset import ClitResultDataset
from clit_recommender.domain.datasets import Dataset, DatasetSplitType
from clit_recommender.config import Config
from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.clit_mock.graph import Graph
from clit_recommender.domain.metrics import Metrics
from clit_recommender.domain.systems import System
from clit_recommender.process.evaluation import Evaluation


class SingleSystem:
    _config: Config
    _data: List[DataRow]
    _save: bool
    _result_dir: str

    def __init__(self, datasets: List[Dataset], result_dir=None) -> None:
        self._config = Config(depth=1, datasets=datasets, results_dir=result_dir)

        self._data = ClitResultDataset(self._config, split_type=DatasetSplitType.EVAL)

        self._save = result_dir is not None

    def run_all(self):
        path = os.path.join(
            self._config.results_dir,
            "single_system",
            "_".join(map(attrgetter("label"), self._config.datasets)),
        )
        for system in self._config.systems:
            print("Starting System", system)
            self.run(system, path)

    def run(self, system: System, path: str = ""):
        config = self._config

        config.experiment_name = f"{system.label.split('/')[-1]}-{int(time())}"
        config.systems = [system]

        if self._save:
            path = os.path.join(path, config.experiment_name)

            os.makedirs(path)

            with open(os.path.join(path, "config.json"), "w") as f:
                f.write(config.to_json())

        last_level = [0.0] * self._config.md_modules_count
        last_level[system.index] = 1.0

        g = Graph.create_1_dim(config, [], last_level, 0)

        row: DataRow
        metrics = Metrics.zeros()
        for (row,) in tqdm(self._data):
            result = g.forward(row)
            metrics += Evaluation.evaluate_results(set(row.actual), result, soft=True)

        print("System", system)
        print(metrics.get_summary())

        if self._save:
            with open(os.path.join(path, "summary.txt"), "w") as f:
                f.write(metrics.get_summary())

            with open(os.path.join(path, "metrics.json"), "w") as f:
                f.write(metrics.to_json())

        return metrics


if __name__ == "__main__":
    single_system = SingleSystem(list(Dataset), save=False)
    single_system.run_all()
