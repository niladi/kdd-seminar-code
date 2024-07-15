from typing import Dict, List, Tuple, Union
import torch


from clit_recommender.models.clit_mock import Graph
from clit_recommender.models.factory import model_factory


from clit_recommender.data.offline_data import OfflineData
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from clit_recommender.data.dataset import DataRow
from clit_recommender.models.base import ClitRecommenderModel, ModelResult

from clit_recommender.config import Config


class ClitRecommeder:
    _config: Config
    _model: ClitRecommenderModel
    _precompute_embeddings: torch.tensor
    _uri_to_idx: LmdbImmutableDict

    def __init__(self, config_or_config_file: Union[str, Config]) -> None:
        if type(config_or_config_file) is str:
            raise NotImplementedError()
        else:
            self._config = config_or_config_file

        self._model = model_factory(self._config)
        offline_data = OfflineData(self._config)
        self._precompute_embeddings = offline_data.load_embeddings()
        self._uri_to_idx = offline_data.load_uri_to_idx()

    def get_model(self) -> ClitRecommenderModel:
        return self._model

    def process_batch(self, data_row: DataRow) -> ModelResult:
        embeddings = self._precompute_embeddings[
            self._uri_to_idx.get(data_row.context_uri)
        ]
        return self._model(embeddings, data_row)
