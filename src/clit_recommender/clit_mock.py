# %%

from typing import List
from abc import ABC, abstractmethod
from copy import deepcopy
import pandas as pd
import itertools
import numpy
from tqdm.auto import tqdm
from util import flat_map

from functools import lru_cache

THRESHOLD = 1


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
    def is_active(self) -> bool:
        # TDOD Threshold
        return max(self.value) >= THRESHOLD

    def operation(self, data: float):
        super().operation(data)


class CombinedNode(Node, ABC):
    input: List[Node]

    def __init__(self, name: str, value: str, input: List[Node]) -> None:
        super().__init__(name, value)
        self.input = input

    def is_active(self) -> bool:
        return max(map(self.input, lambda x: x.value)) >= THRESHOLD

    def calc_on_input(self, data):
        return list(
            filter(lambda x: x is not None, map(lambda x: x.calc(data), self.input))
        )

    @abstractmethod
    def operation(self, data):
        raise NotImplementedError()


class UnionNode(CombinedNode):
    def operation(self, data: float):
        return flat_map(lambda x: x, self.calc_on_input(data))


class IntersectionNode(CombinedNode):
    def operation(self, data: float):
        return set.intersection(*map(set, self.calc_on_input(data)))


class MajorityVoting(CombinedNode):
    def operation(self, data: float):
        res = self.calc_on_input(data)
        min_size = int(len(res) / 2) + 1
        majority_votes = {}
        for item in res:
            if item in majority_votes:
                majority_votes[str(item)] += 1
            else:
                majority_votes[str(item)] = 1
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

    def forward(self, input: List[float]):
        last_level = self.levels[-1]
        for i in last_level.OuputNodes():
            if i.is_active():
                i.operation(input)

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
                matrix.append(
                    [
                        l.union.input[i_n].value,
                        l.intersection.input[i_n].value,
                        l.majority_voting.input[i_n].value,
                    ]
                )
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
                    "UNION": m[0],
                    "INTERS": m[1],
                    "MAJORITY": m[2],
                }
                data.append(row)
        df = pd.DataFrame(data)
        return df

    def from_matrix(input_size: int, depth: int, matrix: List[List[float]]):
        g = Graph(depth, [InputNode("MD" + str(i), 0) for i in range(input_size)])
        for i_l, l in enumerate(g.levels):
            for i_n, n in enumerate(l.input):
                m = matrix[i_l + i_n]
                l.union.input[i_n].value = m[0]
                l.intersection.input[i_n].value = m[1]
                l.majority_voting.input[i_n].value = m[2]
        return g


# %%

input_size = 10
depth = 3

g = Graph(depth, [InputNode("MD" + str(i), 0) for i in range(input_size)])
g.to_dataframe()

# %%


size = 0
for n in range(depth):
    size += input_size + n * 3

matrix = []

valid_graphes = []
permutations = 2 ** (size * 3)
i = 0

pbar = tqdm(total=permutations)
while i < permutations:
    numpy.binary_repr(i, width=size * 3)
    pbar.update(1)
    i += 1

for p in tqdm(range()):
    res = [int(i) for i in bin(p)[2:]]
    matrix = [[res[j] for j in range(k, k + 3)] for k in range(0, size, 3)]
    g = Graph.from_matrix(input_size, depth, matrix)
    if g.valid():
        valid_graphes.append(g)


# %%
