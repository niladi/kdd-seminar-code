# %%

import json
from dataclasses import dataclass
from multiprocessing import freeze_support
from os.path import join
from typing import Dict, Iterator, List, Self


from torch import Tensor
from torch.utils.data import IterableDataset

from clit_recommender.data.best_graphs import BestGraph
from clit_recommender.domain.datasets import Dataset
from clit_recommender import BEST_GRAPHS_PATH, GraphPresentation
from clit_recommender.config import BEST_GRAPHS_JSON_FILE, BEST_GRAPHS_LMDB_FILE, Config
from clit_recommender.data.graph_db_wrapper import ACTUAL_KEY, GraphDBWrapper
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from clit_recommender.domain.clit_result import Mention
from clit_recommender.domain.systems import System

EVAL_SIZE = 800


@dataclass
class DataRow:
    context_uri: str
    context_text: str
    results: List[List[Mention]]
    actual: List[Mention]

    def __hash__(self) -> int:
        return hash(self.context_uri)


@dataclass
class DataRowWithBestGraph(DataRow):
    best_graph: GraphPresentation

    @classmethod
    def from_data_row(cls, row: DataRow, best_graph: GraphPresentation) -> Self:
        return cls(
            row.context_uri, row.context_text, row.results, row.actual, best_graph
        )

    def __hash__(self) -> int:
        return hash((self.context_uri, self.best_graph))


class ClitResultDataset(IterableDataset):
    _index_map: Dict[str, int]
    _config: Config
    _graph_db_wrapper: GraphDBWrapper

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._graph_db_wrapper = GraphDBWrapper(config.datasets)
        self._index_map = config.get_index_map()
        self._config = config

    def _create_data_row(self, uri: str, text: str, actual: List[Mention]) -> DataRow:
        return DataRow(uri, text, [None] * len(list(System)), actual)

    def __iter__(self) -> Iterator[List[DataRow]]:
        offset = 0
        limit = self._config.batch_size
        while True:
            res = self._graph_db_wrapper.get_contexts(limit=limit, offset=offset)
            if res is None or len(res) == 0:
                break
            batch = []
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
                batch.append(data_row)
            yield batch
            offset += self._config.batch_size

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """
        count = self._graph_db_wrapper.get_count()
        return int(count / self._config.batch_size) + 1

    def __getitem__(self, index):
        for i, x in enumerate(self):
            if i == index:
                return x

        raise IndexError()


class ClitRecommenderDataSet(ClitResultDataset):
    _lmdb: LmdbImmutableDict

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        best_graph = BestGraph.from_config(config)
        if not best_graph.exists():
            print("Best Graphs not found. Creating...")
            best_graph.create()
        self._lmdb = best_graph.load_lmdb()

    def __iter__(self) -> Iterator[List[DataRowWithBestGraph]]:
        batch = []
        for data_row in super().__iter__():
            for row in data_row:
                for best_graph in self._lmdb[row.context_uri]:

                    best_graph_tuple: GraphPresentation = Tensor(best_graph)

                    batch.append(
                        DataRowWithBestGraph.from_data_row(row, best_graph_tuple)
                    )
                    if len(batch) == self._config.batch_size:
                        yield batch
                        batch = []
                    elif len(batch) > self._config.batch_size:
                        raise AssertionError("Batch size exceeded")
        if len(batch) > 0:
            yield batch

    def __len__(self) -> int:

        best_graphs = BestGraph.from_config(self._config).load_dict()
        count = sum(map(len, best_graphs.values()))
        return int(count / self._config.batch_size) + 1


if __name__ == "__main__":
    freeze_support()
    c = ClitRecommenderDataSet(Config(batch_size=4, datasets=[Dataset.MED_MENTIONS]))
    print(c._index_map)
