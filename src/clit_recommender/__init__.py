from typing import Tuple, TypeAlias, Union

from torch import Tensor, manual_seed

from random import seed

_seed = 500
print("Set Manual Seed to", _seed)
seed(_seed)
manual_seed(_seed)


DATA_PATH = "/Users/niladi/workspace/seminar-kdd/code/data"
MD_ONLY = f"{DATA_PATH}/2024-05-14/MDOnly"
MD_ONLY_MEDMENTION_PATH = f"{DATA_PATH}/2024-07-28/MDOnly"
MEDMENTION_PATH = f"{DATA_PATH}/medmentions"
DATASETS_PATH = f"{DATA_PATH}/datasets"
CLIT_RESULTS_PATH = f"{DATA_PATH}/clit_results"
BEST_GRAPHS_PATH = f"{DATA_PATH}/best_graphs"
EMBEDDINGS_PATH = f"{DATA_PATH}/embeddings"


GraphPresentation: TypeAlias = Union[Tuple[Tuple[float, float, float], ...], Tensor]
