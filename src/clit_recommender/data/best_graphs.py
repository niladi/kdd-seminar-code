import itertools
import json
from multiprocessing import freeze_support
from os.path import join
from typing import List


import torch
from domain.metrics import Metrics
from pqdm.processes import pqdm
from process.evaluation import Evaluation
from tqdm.auto import tqdm

from clit_recommender.config import Config, BEST_GRAPHS_JSON_FILE, BEST_GRAPHS_LMDB_FILE
from clit_recommender.data.dataset import ClitRecommenderDataset, DataRow
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict
from clit_recommender.models.clit_mock import (
    Graph,
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)


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


def get_best_graph(row: DataRow, graphs: List[Graph]):
    best_results = []
    current_metric: Metrics = Metrics.zeros()
    current_size = 0
    actual = set(row.actual)
    for graph in graphs:
        res = graph.forward(row)
        if res is None or len(res) == 0:
            continue

        metrics = Evaluation.evaluate(actual, set(res))
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


def process_batch(batch: List[DataRow], graphs: List[Graph]):
    best_graph = {}
    for row in batch:
        if row.context_uri in best_graph:
            raise ValueError("Duplicate context_uri")
        best_graph[row.context_uri] = list(
            map(lambda g: g.to_matrix(), get_best_graph(row, graphs))
        )
    return best_graph


if __name__ == "__main__":

    freeze_support()

    # Combine tensor lists using itertools
    _combined_tensors = itertools.product(
        [[]], generate_tensors(10), [IntersectionNode, UnionNode, MajorityVoting]
    )

    _graphs = []
    _c = Config(depth=1, md_modules_count=10, load_best_graph=False)

    for _combined_tensor in tqdm(_combined_tensors):
        _graphs.append(
            Graph.create_1_dim(
                _c,
                _combined_tensor[0],
                _combined_tensor[1],
                _combined_tensor[2],
            )
        )

    _best_graph = {}
    _row: DataRow
    _args = itertools.product(ClitRecommenderDataset(_c), [_graphs])
    _result = pqdm(
        _args,
        process_batch,
        n_jobs=8,
        argument_type="args",
    )

    for _batch_result in _result:
        for _key, _value in _batch_result.items():
            if _key in _best_graph:
                raise ValueError("Duplicate context_uri")
            _best_graph[_key] = _value

    # Dump _best_graph to JSON
    with open(join(_c.cache_dir, BEST_GRAPHS_JSON_FILE), "w") as f:
        json.dump(_best_graph, f)

    LmdbImmutableDict.from_dict(_best_graph, join(_c.cache_dir, BEST_GRAPHS_LMDB_FILE))
