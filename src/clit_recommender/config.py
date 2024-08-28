from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Type
from dataclasses_json import dataclass_json
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.systems import System


@dataclass_json
@dataclass
class Config:
    epochs: int = 2
    batch_size: int = 1
    md_modules_count: int = 14
    depth: int = 1
    results_dir: str = "/Users/niladi/workspace/seminar-kdd/code/data/results"
    lm_model_name: str = "roberta-large"
    device = "cpu"
    experiment_name: str = None
    model: str = "ClitRecommenderModelOneDepth"
    load_best_graph: bool = True
    threshold: int = 0.5
    seed: Optional[int] = 500
    datasets: Optional[List[Dataset]] = None
    systems: Optional[List[System]] = None
    eval_factor: int = 0.15

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"Clit-Recommender-Experiment-{int(time())}"

        if self.datasets is None:
            self.datasets = list(Dataset)

        if self.systems is None:
            self.systems = list(System)

    def calculate_output_size(self) -> int:
        i = self.md_modules_count
        j = self.depth
        return int((j * (2 * i + 3 * j - 3) * 3) / 2)

    def get_index_map(self) -> Dict:
        return {s.uri: s.index for s in self.systems}


BEST_GRAPHS_LMDB_FILE = "best_graphs.lmdb"
BEST_GRAPHS_JSON_FILE = "best_graphs.json"
