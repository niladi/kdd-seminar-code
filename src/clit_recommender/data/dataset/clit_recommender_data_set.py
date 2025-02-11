from clit_recommender import GraphPresentation
from clit_recommender.config import Config

from clit_recommender.data.best_graphs.io import BestGraphIO
from clit_recommender.data.dataset.clit_result_dataset import ClitResultDataset
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict


from torch import Tensor


from typing import Iterator, List, Self


from clit_recommender.domain.data_row import DataRow, DataRowWithBestGraph


from dataclasses import dataclass

from clit_recommender.domain.datasets import Dataset, DatasetSplitType


class ClitRecommenderDataSet(ClitResultDataset):
    # Same results as clit_result_dataset and embeddings (LMDB - a kind of datastore)
    _lmdb: LmdbImmutableDict

    def __init__(
        self, config: Config, split_type: DatasetSplitType = DatasetSplitType.ALL
    ) -> None:
        super().__init__(config, split_type)
        best_graph = BestGraphIO(config)
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

        best_graphs = BestGraphIO(self._config).load_dict()
        count = sum(map(len, best_graphs.values()))
        return int(count / self._config.batch_size) + 1


class ClitRecommenderDynamicBatchDataSet(ClitResultDataset):
    # 
    _lmdb: LmdbImmutableDict

    def __init__(
        self, config: Config, split_type: DatasetSplitType = DatasetSplitType.ALL
    ) -> None:
        super().__init__(config, split_type)
        best_graph = BestGraphIO(config)
        if not best_graph.exists():
            raise AssertionError("The best grpah should exists")

        assert config.batch_size is None or config.batch_size == 1

        print("Warning Batch size is dynamic and depends on the amount of best graphs")

        self._lmdb = best_graph.load_lmdb()

    def __iter__(self) -> Iterator[List[DataRowWithBestGraph]]:
        batch = []
        for data_row_batch in super().__iter__():
            for data_row in data_row_batch:
                for best_graph in self._lmdb[data_row.context_uri]:
                    best_graph_tuple: GraphPresentation = Tensor(best_graph)
                    batch.append(
                        DataRowWithBestGraph.from_data_row(data_row, best_graph_tuple)
                    )
            if len(batch) == 0:
                print("Skipping", data_row.context_uri)
            else:
                yield batch
                batch = []


if __name__ == "__main__":
    list(
        ClitRecommenderDynamicBatchDataSet(
            Config(datasets=[Dataset.RSS_500, Dataset.REUTERS_128])
        )
    )
