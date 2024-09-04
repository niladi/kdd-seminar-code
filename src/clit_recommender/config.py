from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Type
from dataclasses_json import config, dataclass_json
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.metrics import MetricType
from clit_recommender.domain.systems import System
from clit_recommender.util import enum_list_default


@dataclass_json
@dataclass
class Config:
    epochs: int = 2
    batch_size: int = 1
    depth: int = 1
    results_dir: str = "/Users/niladi/workspace/seminar-kdd/code/data/results"
    lm_model_name: str = "roberta-large"
    device: str = "cpu"
    experiment_name: str = None
    model: str = "ClitRecommenderModelOneDepth"
    load_best_graph: bool = True
    threshold: int = 0.5
    seed: Optional[int] = 500
    datasets: Optional[List[Dataset]] = enum_list_default(Dataset)
    systems: Optional[List[System]] = enum_list_default(System)
    metric_type: MetricType = MetricType.F1
    eval_factor: int = 0.15
    md_modules_count: int = len(list(System))

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = (
                f"Clit-Recommender-Experiment-{int(time())}-{self.metric_type}"
            )

        if self.datasets is None or len(self.datasets) == 0:
            self.datasets = list(Dataset)

        if self.systems is None or len(self.systems) == 0:
            self.systems = list(System)

    def calculate_output_size(self) -> int:
        i = self.md_modules_count
        j = self.depth
        return int((j * (2 * i + 3 * j - 3) * 3) / 2)

    def get_index_map(self) -> Dict:
        return {s.uri: s.index for s in self.systems}


BEST_GRAPHS_LMDB_FILE = "best_graphs.lmdb"
BEST_GRAPHS_JSON_FILE = "best_graphs.json"
