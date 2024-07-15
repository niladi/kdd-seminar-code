# %%

from dataclasses import dataclass

from typing import Dict, Iterator, List, Tuple, Union
from os.path import join
import json

from config import BEST_GRAPHS_JSON_FILE, BEST_GRAPHS_LMDB_FILE

from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from torch import Tensor

from clit_recommender.domain.clit_result import Mention
from clit_recommender.data.graph_db_wrapper import ACTUAL_KEY, GraphDBWrapper
from clit_recommender.config import Config

from torch.utils.data import IterableDataset


@dataclass
class DataRow:
    context_uri: str
    context_text: str
    results: List[List[Mention]]
    actual: List[Mention]
    best_graph: Union[Tuple[Tuple[float]], Tensor] = None

    def __hash__(self) -> int:
        return hash(self.context_uri)


class ClitRecommenderDataset(IterableDataset):
    _index_map: Dict[str, int]
    _config: Config
    _graph_db_wrapper: GraphDBWrapper
    _start: int
    _end: int
    _lmdb: LmdbImmutableDict

    def __init__(self, config: Config, start=0, end=None) -> None:
        super().__init__()

        self._graph_db_wrapper = GraphDBWrapper()
        self._index_map = {
            s: i for i, s in enumerate(sorted(self._graph_db_wrapper.get_systems()))
        }
        self._config = config
        self._start = start
        self._end = end
        if self._config.load_best_graph:
            self._lmdb = LmdbImmutableDict(
                join(config.cache_dir, BEST_GRAPHS_LMDB_FILE)
            )

    def _create_data_row(self, uri: str, text: str, actual: List[Mention]) -> DataRow:
        return DataRow(uri, text, [None] * len(self._index_map), actual)

    def __iter__(self) -> Iterator[List[DataRow]]:
        offset = self._start
        batch = []

        limit = self._config.batch_size
        while True:
            if self._end is not None and offset + limit >= self._end:
                limit = self._end - offset
            res = self._graph_db_wrapper.get_contexts(limit=limit, offset=offset)
            if res is None or len(res) == 0:
                break

            for uri, text in res:
                mentions = self._graph_db_wrapper.get_mentions_of_context(uri)
                if ACTUAL_KEY in mentions:
                    actual = mentions.pop(ACTUAL_KEY)
                else:
                    print("No actual mentions found for context", uri)
                    actual = []
                data_row = self._create_data_row(uri, text, actual)
                for system, mention in mentions.items():
                    data_row.results[self._index_map[system]] = mention
                if self._config.load_best_graph:
                    for g in self._lmdb.get(uri, []):
                        data_row.best_graph = g
                        batch.append(data_row)
                        if len(batch) >= self._config.batch_size:
                            yield batch
                            batch = []

                else:
                    batch.append(data_row)
                    if len(batch) >= self._config.batch_size:
                        yield batch
                        batch = []

                offset += 1

            if self._end is not None and offset >= self._end:
                if len(batch) > 0:
                    yield batch
                    batch = []
                break

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """

        if self._config.load_best_graph:
            with open(join(self._config.cache_dir, BEST_GRAPHS_JSON_FILE), "r") as f:
                best_graphs = json.load(f)
            count = sum(map(len, best_graphs.values()))
        else:
            count = self._graph_db_wrapper.get_count()

        if self._end is not None:
            end = min(count, self._end)
            return int((end - self._start) / self._config.batch_size) + 1
        return int((count - self._start) / self._config.batch_size) + 1

    def __getitem__(self, index):
        for i, x in enumerate(self):
            if i == index:
                return x

        raise IndexError()


if __name__ == "__main__":
    l = list(
        ClitRecommenderDataset(Config(batch_size=4, load_best_graph=False), end=10)
    )
    print(len(l))
