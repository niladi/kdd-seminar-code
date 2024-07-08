# %%

from dataclasses import dataclass

from typing import Dict, Iterator, List

from domain.clit_result import Mention
from data.graph_db_wrapper import ACTUAL_KEY, GraphDBWrapper
from clit_recommender.config import Config

from torch.utils.data import IterableDataset


@dataclass
class DataRow:
    context_uri: str
    context_text: str
    results: List[List[Mention]]
    actual: List[Mention]

    def __hash__(self) -> int:
        return hash(self.context_uri)


class ClitRecommenderDataset(IterableDataset):
    _index_map: Dict[str, int]
    _config: Config
    _graph_db_wrapper: GraphDBWrapper
    _start: int
    _end: int

    def __init__(self, config: Config, start=0, end=None) -> None:
        super().__init__()

        self._graph_db_wrapper = GraphDBWrapper()
        self._index_map = {
            s: i for i, s in enumerate(sorted(self._graph_db_wrapper.get_systems()))
        }
        self._config = config
        self._start = start
        self._end = end

    def _create_data_row(self, uri: str, text: str, actual: List[Mention]) -> DataRow:
        return DataRow(uri, text, [None] * len(self._index_map), actual)

    def __iter__(self) -> Iterator[List[DataRow]]:
        offset = self._start
        limit = self._config.batch_size
        while True:
            if self._end is not None and offset + limit >= self._end:
                limit = self._end - offset
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
            if self._end is not None and offset >= self._end:
                break

    def __len__(self) -> int:
        """
        Returns number of batch elements.
        :return: Number of batch elements.
        """
        if self._end is not None:
            return int((self._end - self._start) / self._config.batch_size) + 1
        return (
            int(
                (self._graph_db_wrapper.get_count() - self._start)
                / self._config.batch_size
            )
            + 1
        )

    def __getitem__(self, index):
        for i, x in enumerate(self):
            if i == index:
                return x

        raise IndexError()
