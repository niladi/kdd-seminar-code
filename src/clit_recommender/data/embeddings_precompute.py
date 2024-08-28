# %%

from typing import Dict, List
from os.path import join, exists
from os import remove
import json

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


from clit_recommender.data.dataset import ClitResultDataset, DataRow

from clit_recommender import EMBEDDINGS_PATH
from clit_recommender.config import Config
from clit_recommender.util import flat_map
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict

#


class EmbeddingsPrecompute:
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

        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        row: DataRow
        for row in tqdm(flat_map(lambda x: x, ClitResultDataset(self._config))):
            idx = uri_to_idx.get(row.context_uri)
            inputs = tokenizer(row.context_text, return_tensors="pt")

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings[idx] = outputs.last_hidden_state.mean(dim=1)

        filename = join(EMBEDDINGS_PATH, self.embeddings_filname)
        if exists(filename):
            remove(filename)

        torch.save(embeddings, filename)

    def generate_uri_to_idx(self):

        uri_to_idx: Dict[str, int] = dict()

        index: int
        row: DataRow
        for index, row in enumerate(
            flat_map(lambda x: x, ClitResultDataset(self._config))
        ):
            uri_to_idx[row.context_uri] = index

        p = join(EMBEDDINGS_PATH, self.uri_to_idx_filename_lmdb)

        if exists(p):
            remove(p)

        LmdbImmutableDict.from_dict(uri_to_idx, p)

        with open(join(EMBEDDINGS_PATH, self.uri_to_idx_filename_json), "w") as f:
            json.dump(uri_to_idx, f)

    def load_uri_to_idx(self) -> LmdbImmutableDict:
        return LmdbImmutableDict(join(EMBEDDINGS_PATH, self.uri_to_idx_filename_lmdb))

    def load_embeddings(self) -> torch.tensor:
        return torch.load(join(EMBEDDINGS_PATH, self.embeddings_filname))


if __name__ == "__main__":
    _config = Config()
    offline_data = EmbeddingsPrecompute(_config)
    offline_data.generate_uri_to_idx()
    offline_data.generate_text_embeddings()
    print("Done")
