# %%

from typing import List, Type, Union
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd

from torch import Tensor

from clit_recommender.dataset import DataRow
from clit_recommender.util import flat_map
from clit_recommender.config import Config
from clit_recommender.clit_result import Mention

from functools import lru_cache

THRESHOLD = 0.7


class Node(ABC):
    name: str
    value: float

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value

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

    def __init__(self, name: str, value: str, system_index: int) -> None:
        super().__init__(name, value)
        self.system_index = system_index

    def is_active(self) -> bool:
        # TDOD Threshold
        return self.value >= THRESHOLD

    def operation(self, data: DataRow):
        return data.results[self.system_index]


class CombinedNode(Node, ABC):
    input: List[Node]

    def __init__(self, name: str, value: str, input: List[Node]) -> None:
        super().__init__(name, value)
        self.input = input

    def is_active(self) -> bool:
        return max(map(lambda x: x.value, self.input)) >= THRESHOLD

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
        return flat_map(lambda x: x, self.calc_on_input(data))


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
        return majority_voted


class Level:
    input: List[Node]
    majority_voting: MajorityVoting
    intersection: IntersectionNode
    union: UnionNode

    def __init__(self, level_name: str, nodes: List[Node]):
        self.input = nodes
        self.majority_voting = MajorityVoting(
            level_name + "_majority_voting", 0, deepcopy(nodes)
        )
        self.intersection = IntersectionNode(
            level_name + "_intersection", 0, deepcopy(nodes)
        )
        self.union = UnionNode(level_name + "_union", 0, deepcopy(nodes))

    def OuputNodes(self) -> List[Node]:
        return deepcopy(self.input) + [
            self.majority_voting,
            self.intersection,
            self.union,
        ]


class Graph:
    levels: List[Level]
    intput_size: int

    def __init__(self, depth: int, input_node: List[InputNode]) -> None:
        self.levels = []
        self.input_size = len(input_node)
        for i in range(depth):
            self.levels.append(Level("Level_" + str(i), input_node))
            input_node = self.levels[-1].OuputNodes()

    def forward(self, data_row: DataRow) -> List[Mention]:
        last_level = self.levels[-1]
        for i in last_level.OuputNodes():
            if i.is_active():
                return set(i.calc(data_row))

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

    def to_matrix(self) -> List[List[float]]:
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
        return matrix

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
    def create(config: Config, value_matrix: Union[List[List[float]], Tensor] = None):
        depth = config.depth
        input_size = config.md_modules_count

        g = Graph(depth, [InputNode("MD" + str(i), 0, i) for i in range(input_size)])

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

    @staticmethod
    def create_by_last_as_vector_and_label(
        config: Config,
        value_matrix: Union[List[List[float]], Tensor],
        last_level_values: Union[List[float], Tensor],
        last_level_type: Type[CombinedNode],
    ):
        assert int(config.calculate_output_size() / 3 - len(value_matrix)) == len(
            last_level_values
        )

        new_matrix = deepcopy(value_matrix)

        for v in last_level_values:
            row = [0] * 3
            row[last_level_type.get_index()] = v
            new_matrix.append(row)

        return Graph.create(config, new_matrix)
