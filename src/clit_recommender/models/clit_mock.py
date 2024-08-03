from typing import List, Set, Tuple, Type, Union
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import numpy as np

from torch import Tensor

from clit_recommender import GraphPresentation
from clit_recommender.data.dataset import DataRow
from clit_recommender.util import flat_map
from clit_recommender.config import Config
from clit_recommender.domain.clit_result import Mention

from functools import lru_cache


class Node(ABC):
    name: str
    value: float
    threshold: float

    def __init__(self, name: str, value: str, threshold: float) -> None:
        self.name = name
        self.value = value
        self.threshold = threshold

    @lru_cache(maxsize=None)
    def calc(self, data):
        if self.is_active():
            return self.operation(data)
        return None

    @lru_cache(maxsize=None)
    @abstractmethod
    def is_active(self) -> bool:
        raise NotImplementedError()

    @lru_cache(maxsize=None)
    @abstractmethod
    def operation(self, data):
        raise NotImplementedError()


class InputNode(Node):
    system_index: int

    def __init__(
        self, name: str, value: str, threshold: float, system_index: int
    ) -> None:
        super().__init__(name, value, threshold)
        self.system_index = system_index

    def is_active(self) -> bool:
        # TDOD Threshold
        return self.value >= self.threshold

    def operation(self, data: DataRow):
        return data.results[self.system_index]


class CombinedNode(Node, ABC):
    input: List[Node]

    def __init__(
        self, name: str, value: str, threshold: float, input: List[Node]
    ) -> None:
        super().__init__(name, value, threshold)
        self.input = input

    def is_active(self) -> bool:
        return max(map(lambda x: x.value, self.input)) >= self.threshold

    def calc_on_input(self, data: DataRow):
        return list(
            filter(lambda x: x is not None, map(lambda x: x.calc(data), self.input))
        )

    @abstractmethod
    def operation(self, data):
        raise NotImplementedError()

    @abstractmethod
    def get_index() -> int:
        raise NotImplementedError()


class UnionNode(CombinedNode):
    def get_index() -> int:
        return 0

    def operation(self, data: float):
        return set(flat_map(lambda x: x, self.calc_on_input(data)))


class IntersectionNode(CombinedNode):
    def get_index() -> int:
        return 1

    def operation(self, data: float):
        inputs = list(map(set, self.calc_on_input(data)))
        if inputs is None or len(inputs) == 0:
            return set()
        return set.intersection(*inputs)


class MajorityVoting(CombinedNode):
    def get_index() -> int:
        return 2

    def operation(self, data):
        res = self.calc_on_input(data)
        min_size = int(len(res) / 2) + 1
        majority_votes = {}
        for mentions in res:
            for mention in mentions:
                if mention in majority_votes:
                    majority_votes[mention] += 1
                else:
                    majority_votes[mention] = 1
        majority_voted = [
            item for item, count in majority_votes.items() if count >= min_size
        ]
        return set(majority_voted)


class Level:
    input: List[Node]
    majority_voting: MajorityVoting
    intersection: IntersectionNode
    union: UnionNode

    def __init__(self, level_name: str, threshold: float, nodes: List[Node]):
        self.input = nodes
        self.majority_voting = MajorityVoting(
            level_name + "_majority_voting", 0, threshold, deepcopy(nodes)
        )
        self.intersection = IntersectionNode(
            level_name + "_intersection", 0, threshold, deepcopy(nodes)
        )
        self.union = UnionNode(level_name + "_union", 0, threshold, deepcopy(nodes))

    def OuputNodes(self) -> List[Node]:
        return deepcopy(self.input) + [
            self.majority_voting,
            self.intersection,
            self.union,
        ]


class Graph:
    levels: List[Level]
    intput_size: int
    threshold: float

    def __init__(
        self, depth: int, threshold: float, input_node: List[InputNode]
    ) -> None:
        self.levels = []
        self.input_size = len(input_node)
        self.threshold = threshold
        for i in range(depth):
            self.levels.append(Level("Level_" + str(i), threshold, input_node))
            input_node = self.levels[-1].OuputNodes()

    def forward(self, data_row: DataRow) -> Set[Mention]:
        level = self.get_last_level_node()
        return set() if level is None else set(level.calc(data_row))

    def valid(self) -> bool:
        last_level = self.levels[-1]
        one_is_active = False
        for i in last_level.OuputNodes():
            if i.is_active():
                if one_is_active:
                    return False
                else:
                    one_is_active = True
        return one_is_active

    def to_matrix(self) -> GraphPresentation:
        matrix = []
        for i_l, l in enumerate(self.levels):
            for i_n, n in enumerate(l.input):
                row = [None] * 3
                row[UnionNode.get_index()] = float(l.union.input[i_n].value)
                row[IntersectionNode.get_index()] = float(
                    l.intersection.input[i_n].value
                )
                row[MajorityVoting.get_index()] = float(
                    l.majority_voting.input[i_n].value
                )
                matrix.append(row)
        return tuple(map(tuple, matrix))

    def to_matrix_rounded(self) -> GraphPresentation:
        matrix = np.matrix(self.to_matrix())
        matrix[matrix >= self.threshold] = 1.0
        matrix[matrix < self.threshold] = 0.0
        return tuple(map(tuple, matrix.tolist()))

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        matrix = self.to_matrix()
        for i_l, l in enumerate(self.levels):
            for i_n, n in enumerate(l.input):
                m = matrix[i_l + i_n]
                row = {
                    "LEVEL": i_l,
                    "INPUT": n.name,
                    "UNION": m[UnionNode.get_index()],
                    "INTERS": m[IntersectionNode.get_index()],
                    "MAJORITY": m[MajorityVoting.get_index()],
                }
                data.append(row)
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def create(
        config: Config,
        value_matrix: GraphPresentation = None,
    ):
        depth = config.depth
        input_size = config.md_modules_count
        threshold = config.threshold

        g = Graph(
            depth,
            threshold,
            [InputNode("MD" + str(i), 0, threshold, i) for i in range(input_size)],
        )

        if value_matrix is None:
            return g

        i = 0
        for i_l, l in enumerate(g.levels):
            for i_n, n in enumerate(l.input):
                m = value_matrix[i]
                l.union.input[i_n].value = m[UnionNode.get_index()]
                l.intersection.input[i_n].value = m[IntersectionNode.get_index()]
                l.majority_voting.input[i_n].value = m[MajorityVoting.get_index()]
                i += 1
        return g

    def get_last_level_node(self) -> CombinedNode:
        last_level = self.levels[-1]
        for i in last_level.OuputNodes():
            if i.is_active():
                return i

    def get_last_level_tuple(self) -> Tuple[Tuple[float, ...], Type[CombinedNode]]:
        node = self.get_last_level_node()
        return tuple(map(lambda x: x.value, node.input)), type(node)

    @staticmethod
    def create_1_dim(
        config: Config,
        value_matrix: GraphPresentation,
        last_level_values: Union[List[float], Tensor, Tuple[float, ...]],
        last_level_type: Union[Type[CombinedNode], int],
    ):
        assert int(config.calculate_output_size() / 3 - len(value_matrix)) == len(
            last_level_values
        )

        new_matrix = deepcopy(value_matrix)
        index = (
            last_level_type
            if isinstance(last_level_type, int)
            else last_level_type.get_index()
        )
        for v in last_level_values:
            row = [0] * 3
            row[index] = v
            new_matrix.append(row)

        return Graph.create(config, new_matrix)
