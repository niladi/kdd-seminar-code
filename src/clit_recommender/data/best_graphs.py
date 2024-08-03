import itertools
import json
from functools import lru_cache
from multiprocessing import freeze_support
from os import remove
from os.path import exists, join
from typing import Dict, Iterable, List

import torch
import torch.multiprocessing
from pqdm.processes import pqdm
from tqdm.auto import tqdm

from clit_recommender import BEST_GRAPHS_PATH
from clit_recommender.config import BEST_GRAPHS_JSON_FILE, BEST_GRAPHS_LMDB_FILE, Config
from clit_recommender.data.dataset import ClitResultDataset, DataRow
from clit_recommender.data.graph_db_wrapper import GraphDBWrapper
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from clit_recommender.domain.datasets import DatasetEnum
from clit_recommender.domain.metrics import Metrics
from clit_recommender.models.clit_mock import (
    Graph,
    GraphPresentation,
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)


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


def get_best_graph(row: DataRow, graphs: Iterable[Graph]):
    best_results = []
    current_metric: Metrics = Metrics.zeros()
    current_size = 0
    actual = set(row.actual)
    for graph in graphs:
        res = graph.forward(row)
        if res is None or len(res) == 0:
            continue

        metrics = Metrics.evaluate_results(actual, set(res))
        s = sum(map(sum, graph.to_matrix()))

        if metrics.get_f1() == current_metric.get_f1():
            if s < current_size:
                best_results = [graph]
                current_size = s
            elif s == current_size:
                best_results.append(graph)
        elif metrics.get_f1() > current_metric.get_f1():
            best_results = [graph]
            current_metric = metrics
            current_size = s
    return best_results


def process_batch(batch: List[DataRow], graphs: Iterable[Graph]):
    best_graph = {}
    for row in batch:
        if row.context_uri in best_graph:
            raise ValueError("Duplicate context_uri")
        best_graph[row.context_uri] = list(
            map(lambda g: g.to_matrix(), get_best_graph(row, graphs))
        )
    return best_graph


def create_best_graphs(
    _datasets: List[DatasetEnum], _best_graph: Dict, override_existing=False
):

    for dataset in tqdm(_datasets):
        _graph_db_wrapper = GraphDBWrapper([dataset])

        _amount = len(_graph_db_wrapper.get_all_systems())
        used = _graph_db_wrapper.get_systems_on_datasets()
        _c = Config(
            depth=1,
            md_modules_count=_amount,
            load_best_graph=False,
            batch_size=16,
            datasets=[dataset],
        )

        d = ClitResultDataset(_c)
        index = list(map(d._index_map.get, used))
        _t = torch.zeros(_amount)
        _t[index] = 1

        _tensors = generate_tensors(_amount)  # should be cached

        print(len(_tensors))
        _tensors = list(filter(lambda x: x.sum() == (x * _t).sum(), _tensors))
        print(len(_tensors))

        # Combine tensor lists using itertools
        _combined_tensors = itertools.product(
            [[]], _tensors, [IntersectionNode, UnionNode, MajorityVoting]
        )

        _graphs = []

        for _combined_tensor in tqdm(_combined_tensors, total=len(_tensors) * 3):
            _graphs.append(
                Graph.create_1_dim(
                    _c,
                    _combined_tensor[0],
                    _combined_tensor[1],
                    _combined_tensor[2],
                )
            )

        _row: DataRow
        _rows_list = list(d)
        _args = itertools.product(_rows_list, [_graphs])

        _result = pqdm(
            _args,
            process_batch,
            n_jobs=4,
            argument_type="args",
        )

        print(_result[0])

        for _batch_result in _result:
            for _key, _value in _batch_result.items():
                if not override_existing and _key in _best_graph:
                    raise ValueError("Duplicate context_uri")
                _best_graph[_key] = _value

    # Dump _best_graph to JSON
    with open(join(BEST_GRAPHS_PATH, BEST_GRAPHS_JSON_FILE), "w") as f:
        json.dump(_best_graph, f)

    lmdb_path = join(BEST_GRAPHS_PATH, BEST_GRAPHS_LMDB_FILE)

    if exists(lmdb_path):
        remove(lmdb_path)

    LmdbImmutableDict.from_dict(_best_graph, lmdb_path)


if __name__ == "__main__":

    # Because of extention of MedMentions
    with open(join(BEST_GRAPHS_PATH, BEST_GRAPHS_JSON_FILE), "r") as f:
        _best_graph = json.load(f)

    _best_graph: Dict[str, GraphPresentation] = {
        key: list(
            tuple(tuple(float(value) for value in level) for level in graph)
            for graph in graphs
        )
        for key, graphs in _best_graph.items()
    }

    freeze_support()

    # _datasets = list(DatasetEnum)
    _datasets = []

    create_best_graphs(_datasets, _best_graph, True)
