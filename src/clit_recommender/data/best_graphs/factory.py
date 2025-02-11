import itertools
from functools import lru_cache
from multiprocessing import freeze_support
from os import makedirs
from os.path import exists
from typing import Iterable, List


import torch
import torch.multiprocessing
from pqdm.processes import pqdm
from tqdm.auto import tqdm

from clit_recommender.config import Config


from clit_recommender.data.best_graphs.io import BestGraphIO
from clit_recommender.data.dataset.clit_result_dataset import ClitResultDataset
from clit_recommender.domain.data_row import DataRow
from clit_recommender.data.graph_db_wrapper import GraphDBWrapper
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.metrics import MetricType, Metrics
from clit_recommender.domain.systems import System
from clit_recommender.domain.clit_mock.graph import (
    Graph,
)
from clit_recommender.domain.clit_mock.combined_node import (
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)
from clit_recommender.eval.evaluation import Evaluation


class BestGraphFactory(BestGraphIO):

    @staticmethod
    @lru_cache
    def generate_tensors(n):
        def generate_tensor(i):
            binary = bin(i)[2:].zfill(n)
            return torch.tensor([int(bit) for bit in binary])

        end = 2**n
        i = 0
        tensors = []
        with tqdm(total=end) as pbar:
            while i < end:
                tensors.append(generate_tensor(i))
                i += 1
                pbar.update(1)

        return tensors

    def get_best_graph(self, row: DataRow, graphs: Iterable[Graph]):
        best_results = []
        best_metric: Metrics = Metrics.zeros()
        best_size = 0
        actual = set(row.actual)
        for graph in graphs:
            res = graph.forward(row)
            if res is None or len(res) == 0:
                continue

            current_metric = Evaluation.evaluate_results(actual, set(res))
            current_size = sum(map(sum, graph.to_matrix()))

            if current_metric.get_metric(
                self.config.metric_type
            ) == best_metric.get_metric(self.config.metric_type):
                if current_size < best_size:
                    best_results = [graph]
                    best_size = current_size
                elif current_size == best_size:
                    best_results.append(graph)
            elif current_metric.get_metric(
                self.config.metric_type
            ) > best_metric.get_metric(self.config.metric_type):
                best_results = [graph]
                best_metric = current_metric
                best_size = current_size
        return best_results

    def process_batch(self, batch: List[DataRow], graphs: Iterable[Graph]):
        best_graph = {}
        for row in batch:
            if row.context_uri in best_graph:
                raise ValueError("Duplicate context_uri")
            best_graph[row.context_uri] = list(
                map(
                    lambda g: g.to_matrix(),
                    self.get_best_graph(row, graphs),
                )
            )
        return best_graph

    def create(self, load_checkpoint=True, error_on_existing=False):
        best_graph = {}
        if not exists(self.path):
            makedirs(self.path)
        elif load_checkpoint and self.exists():
            best_graph = self.load_dict()

        for dataset in tqdm(self.config.datasets):
            _graph_db_wrapper = GraphDBWrapper([dataset])

            _amount = len(list(System))
            used = _graph_db_wrapper.get_systems_on_datasets()
            _c = Config(
                # depth should always be 1 (>1 would be highly complex interactions)
                depth=1,
                # Has to be initialised false since we are doing it here
                load_best_graph=False,
                batch_size=16,
                # Just for the current dataset
                # Main reason to create it, so it knows what dataset to do it
                datasets=[dataset],
                systems=self.config.systems,
            )
            index_map = _c.get_index_map()
            # ClitResultDataset is dataset without best graph
            d = ClitResultDataset(_c)
            index = []
            # used = systems in use
            for u in used:
                if u in index_map:
                    index.append(index_map.get(u))

            _t = torch.zeros(_amount)
            # Aka. systems we want our recommender to use
            # If we don't want a system to be used, set it to 0, 
            # but best graph must then be recomputed
            _t[index] = 1

            # Generates ALL permutations and then we filter afterwards
            # a single tensor can be: 0111101
            _tensors = self.generate_tensors(_amount)  # should be cached

            print(len(_tensors))
            # Filter out all of the ones we don't want
            # x.sum()
            # _t = which systems we want (e.g. 0, 0, 1, 1, 1)
            # x * _t = element-wise multiplication to set to 0 the ones that are not in the combo 
            # or they are not wanted
            # 
            # The amount of 1s (implicitly 0s) has to be the same before and after the element-wise multiplication
            # 
            # import numpy as np
            # f = np.array([1, 1, 0])  # -> _t
            # x = np.array(
                # [
                    # [0, 0, 0], # -> y
                    # [0, 0, 1],# -> n
                    # [0, 1, 0],# -> y
                    # [0, 1, 1],# -> n
                    # [1, 0, 0],# -> y
                    # [1, 0, 1],# -> n
                    # [1, 1, 0],# -> y
                    # [1, 1, 1],# -> #
                # ]
            # )

            # print(list(filter(lambda y: y.sum() == (f * y).sum(), x)))
            #
            # [array([0, 0, 0]), array([0, 1, 0]), array([1, 0, 0]), array([1, 1, 0])]

            ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            ## IDEA: Set that the sum of the 1s has to be the exact fixed length that we want
            ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Niklas' suggestion: list(filter(lambda y: y.sum() == fixed_size_combo, x))
            # fixed_size_combo = self.config.fixed_size_combo
            _tensors = list(filter(lambda x: x.sum() == (x * _t).sum(), _tensors))
            print(len(_tensors))

            # Combine tensor lists using itertools into permutations
            _combined_tensors = itertools.product(
                [[]], _tensors, [IntersectionNode, UnionNode, MajorityVoting]
            )

            _graphs = []

            for _combined_tensor in tqdm(_combined_tensors, total=len(_tensors) * 3):
                _graphs.append(
                    # this is the mock
                    Graph.create_1_dim(
                        _c,
                        _combined_tensor[0],# empty list
                        _combined_tensor[1],# tensor
                        _combined_tensor[2],# types of aggregation
                    )
                )

            _row: DataRow
            _rows_list = list(d)
            _args = itertools.product(_rows_list, [_graphs])

            _result = pqdm(
                _args,
                self.process_batch,
                n_jobs=4,
                argument_type="args",
            )

            print(_result[0])

            for _batch_result in _result:
                for _key, _value in _batch_result.items():
                    if error_on_existing and _key in best_graph:
                        raise ValueError("Duplicate context_uri")
                    best_graph[_key] = _value

        self.save(best_graph)


if __name__ == "__main__":
    freeze_support()
    for metrics_type in list(MetricType):
        BestGraphFactory(Config(metric_type=metrics_type)).create()
