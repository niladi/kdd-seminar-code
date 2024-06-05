from dataclasses import dataclass
from os import listdir
from typing import Dict, Iterator, List

from clit_result import Mention
from graph_db_wrapper import GraphDBWrapper
from pynif import NIFCollection, NIFContext
from torch.utils.data import IterableDataset

from clit_recommender import DATASETS_PATH


@dataclass
class DataRow:
    context_uri: str
    context_text: str
    results: List[List[Mention]] 


class ClitRecommenderDataset(IterableDataset):
    _index_map: Dict[str, int]
    _batch_size: int
    _graph_db_wrapper: GraphDBWrapper

    def __init__(self) -> None:
        super().__init__()
        self._batch_size = 2
        self._graph_db_wrapper = GraphDBWrapper()
        self._index_map = {
            s: i for i, s in enumerate(sorted(self._graph_db_wrapper.get_systems()))
        }

    def _create_data_row(self, uri: str, text:str)  -> DataRow:
        return DataRow(uri, text, [None] * len(self._index_map))


    def __iter__(self) -> Iterator:
        offset = 0
        while True:
            res = self._graph_db_wrapper.get_contexts(limit=self._batch_size, offset=offset)
            if res == None or len(res) == 0:
                break
            for uri, text in res:
                data_row = self._create_data_row(uri, text)
                mentions = self._graph_db_wrapper.get_mentions_of_context(uri)
                for system, mention in mentions.items():
                    data_row.results[self._index_map[system]] = mention
                yield data_row

    def __getitem__(self, index):
        for i, x in enumerate(self):
            if i == index:
                return x

        raise IndexError()

    def _get_from_sparql(self)
