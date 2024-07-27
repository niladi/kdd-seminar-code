# %%

from typing import Dict, List
from os.path import join
import json

import torch
from transformers import AutoModel, AutoTokenizer


from data.dataset import ClitResultDataset, DataRow

from clit_recommender.config import Config
from clit_recommender.util import flat_map
from data.lmdb_wrapper import LmdbImmutableDict


class OfflineData:
    _config: Config

    uri_to_idx_filename_json = "uri_to_idx.json"
    uri_to_idx_filename_lmdb = "uri_to_idx.lmdb"

    embeddings_filname = "embeddings_tns.pt"

    def __init__(self, config: Config) -> None:
        self._config = config

    def generate_text_embeddings(self):
        uri_to_idx = self.load_uri_to_idx()

        embeddings = [None] * len(uri_to_idx)
        model_name = self._config.lm_model_name
        model_max_length = self._config.lm_hidden_size

        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            return_tensors="pt",
            max_length=model_max_length,
            truncation=True,
            padding="max_length",
        )

        row: DataRow
        for row in flat_map(lambda x: x, ClitResultDataset(self._config)):
            idx = uri_to_idx.get(row.context_uri)
            inputs = tokenizer(
                row.context_text,
                return_tensors="pt",
                max_length=model_max_length,
                truncation=True,
                padding="max_length",
            )

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings[idx] = outputs.last_hidden_state.mean(dim=1)

        torch.save(embeddings, join(self._config.cache_dir, self.embeddings_filname))

    def generate_uri_to_idx(self):

        uri_to_idx: Dict[str, int] = dict()

        index: int
        row: DataRow
        for index, row in enumerate(
            flat_map(lambda x: x, ClitResultDataset(self._config))
        ):
            uri_to_idx[row.context_uri] = index

        LmdbImmutableDict.from_dict(
            uri_to_idx, join(self._config.cache_dir, self.uri_to_idx_filename_lmdb)
        )

        with open(
            join(self._config.cache_dir, self.uri_to_idx_filename_json), "w"
        ) as f:
            json.dump(uri_to_idx, f)

    def load_uri_to_idx(self) -> LmdbImmutableDict:
        return LmdbImmutableDict(
            join(self._config.cache_dir, self.uri_to_idx_filename_lmdb)
        )

    def load_embeddings(self) -> torch.tensor:
        return torch.load(join(self._config.cache_dir, self.embeddings_filname))
