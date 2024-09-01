from clit_recommender.config import Config
from clit_recommender.domain.data_row import DataRow
from clit_recommender.data.graph_db_wrapper import ACTUAL_KEY, GraphDBWrapper
from clit_recommender.domain.clit_result import Mention
from clit_recommender.domain.systems import System


from torch.utils.data import IterableDataset


from typing import Dict, Iterator, List


class ClitResultDataset(IterableDataset):
    _index_map: Dict[str, int]
    _config: Config
    _graph_db_wrapper: GraphDBWrapper

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._graph_db_wrapper = GraphDBWrapper(config.datasets, config.systems)
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
