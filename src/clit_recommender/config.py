from dataclasses import dataclass
from time import time
from typing import List, Optional, Type
from dataclasses_json import dataclass_json
from clit_recommender.domain.datasets import DatasetEnum


@dataclass_json
@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 1
    md_modules_count: int = 14
    depth: int = 2
    cache_dir: str = "/Users/niladi/workspace/seminar-kdd/code/data/cache"
    lm_model_name: str = "roberta-large"
    lm_hidden_size: int = 512
    device = "cpu"
    experiment_name: str = None
    model: str = "ClitRecommenderModelOneDepth"
    load_best_graph: bool = True
    threshold: int = 0.5
    seed: Optional[int] = 500
    datasets: Optional[List[DatasetEnum]] = None
    eval_factor: int = 0.15

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"Clit-Recommender-Experiment-{int(time())}"

        if self.datasets is None:
            self.datasets = list(DatasetEnum)

    def calculate_output_size(self) -> int:
        i = self.md_modules_count
        j = self.depth
        return int((j * (2 * i + 3 * j - 3) * 3) / 2)


BEST_GRAPHS_LMDB_FILE = "best_graphs.lmdb"
BEST_GRAPHS_JSON_FILE = "best_graphs.json"
