from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Literal, Optional, Type
from dataclasses_json import config, dataclass_json
from os.path import join


from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.metrics import MetricType
from clit_recommender.domain.systems import System
from clit_recommender.util import enum_list_default
from clit_recommender import DATA_PATH


@dataclass_json
@dataclass
class Config:
    graphdb_address = "http://localhost:7200/repositories/ReCoLTeDB"
    #http://localhost:7200/repositories/KDD
    # Determine margin range for "best combination" we want to determine "best candidates" with 
    keep_shortest_combos_only = True
    margin_range: float = 0.05
    epochs: int = 20
    batch_size: int = 1
    depth: int = 1
    results_dir: str = join(DATA_PATH, "results")
    lm_model_name: str = "roberta-large"
    device: str = "cpu"
    experiment_name: str = None
    model: str = "ClitRecommenderModelOneDepth"
    load_best_graph: bool = True
    threshold: float = 0.5
    datasets: Optional[List[Dataset]] = enum_list_default(Dataset)
    systems: Optional[List[System]] = enum_list_default(System)
    metric_type: MetricType = MetricType.F1
    md_modules_count: int = len(list(System))
    best_model_eval_type: Literal["result", "prediction"] = "result"
    progess: bool = True
    model_depth: int = 1
    model_hidden_layer_size: int = 512
    fixed_size_combo: int = -1

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = self.create_name()

        if self.datasets is None or len(self.datasets) == 0:
            self.datasets = list(Dataset)

        if self.systems is None or len(self.systems) == 0:
            self.systems = [
                System.BABEFLY,
                System.DBPEDIA_SPOTLIGHT,
                System.OPEN_TAPIOCA,
                System.REFINED_MD_PROPERTIES,
                System.REL_MD_PROPERTIES,
                System.SPACY_MD_PROPERTIES,
                System.TAGME,
                System.TEXT_RAZOR,
            ]

    def create_name(self) -> str:
        return f"ReCoLTe-Experiment-{int(time())}-{self.metric_type.name.lower()}"

    def calculate_output_size(self) -> int:
        i = self.md_modules_count
        j = self.depth
        return int((j * (2 * i + 3 * j - 3) * 3) / 2)

    def get_index_map(self) -> Dict:
        return {s.uri: s.index for s in self.systems}


BEST_GRAPHS_LMDB_FILE = "best_graphs.lmdb"
BEST_GRAPHS_JSON_FILE = "best_graphs.json"
