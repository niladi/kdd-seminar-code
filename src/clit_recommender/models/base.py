from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from torch import Tensor, nn

from clit_recommender.models.clit_mock import Graph
from clit_recommender.config import Config
from clit_recommender.data.dataset import DataRow


@dataclass
class ModelResult:
    logits: Tensor
    loss: Optional[Tensor]


class ClitRecommenderModel(ABC, nn.Module):

    _config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

    @abstractmethod
    def forward(self, embeddings: Tensor, data_row: DataRow) -> ModelResult:
        raise NotImplementedError()
