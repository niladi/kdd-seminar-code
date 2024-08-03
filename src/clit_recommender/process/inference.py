from typing import Union

import torch

from clit_recommender.config import Config
from clit_recommender.data.dataset import DataRowWithBestGraph
from clit_recommender.data.embeddings_precompute import EmbeddingsPrecompute
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from clit_recommender.models.base import ClitRecommenderModel, ModelResult
from clit_recommender.models.clit_mock import Graph
from clit_recommender.models.factory import model_factory


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
        embeddings_data = EmbeddingsPrecompute(self._config)
        self._precompute_embeddings = embeddings_data.load_embeddings()
        self._uri_to_idx = embeddings_data.load_uri_to_idx()

    def get_model(self) -> ClitRecommenderModel:
        return self._model

    def process_batch(self, data_row: DataRowWithBestGraph) -> ModelResult:
        embeddings = self._precompute_embeddings[
            self._uri_to_idx.get(data_row.context_uri)
        ]
        return self._model(embeddings, data_row)
