from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from torch import Tensor, nn
from transformers import AutoConfig

from clit_recommender.models.clit_mock import Graph
from clit_recommender.config import Config
from clit_recommender.data.dataset import DataRow


@dataclass
class ModelResult:
    logits: Tensor
    loss: Optional[Tensor]


class ClitRecommenderModel(ABC, nn.Module):

    _config: Config
    _embedding_size: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._embedding_size = AutoConfig.from_pretrained(
            config.lm_model_name
        ).hidden_size

    @abstractmethod
    def forward(self, embeddings: Tensor, data_row: DataRow) -> ModelResult:
        raise NotImplementedError()
