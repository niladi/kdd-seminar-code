from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor, nn
from transformers import AutoConfig

from clit_recommender.domain.clit_mock.graph import Graph
from clit_recommender.config import Config
from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.model_result import ModelResult


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
