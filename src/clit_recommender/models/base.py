from abc import ABC, abstractmethod
from clit_recommender.config import Config
from data.dataset import DataRow
from torch import Tensor, nn


class ClitRecommenderModel(ABC, nn.Module):

    _config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

    @abstractmethod
    def forward(self, embeddings: Tensor, data_row: DataRow):
        raise NotImplementedError()
