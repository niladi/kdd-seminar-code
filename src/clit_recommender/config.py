from dataclasses import dataclass
from time import time
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 32
    md_modules_count: int = 10
    depth: int = 2
    cache_dir: str = "/Users/niladi/workspace/seminar-kdd/data/cache"
    lm_model_name: str = "roberta-base"
    lm_hidden_size: int = 768
    device = "mps"
    experiment_name: str = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"Clit-Recommender-Experiment-{int(time())}"

    def calculate_output_size(self) -> int:
        i = self.md_modules_count
        j = self.depth
        return int((j * (2 * i + 3 * j - 3) * 3) / 2)
