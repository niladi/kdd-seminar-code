from clit_recommender import GraphPresentation
from clit_recommender.config import Config

from clit_recommender.data.best_graphs.io import BestGraphIO
from clit_recommender.data.dataset.clit_result_dataset import ClitResultDataset
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict


from torch import Tensor


from typing import Iterator, List, Self


from clit_recommender.data.dataset.clit_result_dataset import DataRow


from dataclasses import dataclass


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


class ClitRecommenderDataSet(ClitResultDataset):
    _lmdb: LmdbImmutableDict

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        best_graph = BestGraphIO.from_config(config)
        if not best_graph.exists():
            raise AssertionError("The best grpah should exists")

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

        best_graphs = BestGraphIO.from_config(self._config).load_dict()
        count = sum(map(len, best_graphs.values()))
        return int(count / self._config.batch_size) + 1
