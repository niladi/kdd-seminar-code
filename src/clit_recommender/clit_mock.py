# %%
from turtle import pd
from typing import List
from abc import ABC, abstractmethod
from copy import deepcopy
import pandas as pd


class Node(ABC):
    name: str
    value: float

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value

    @abstractmethod
    def operation(self, data):
        raise NotImplementedError()


class InputNode(Node):

    def operation(self, data: float):
        super().operation(data)


class CombinedNode(Node, ABC):
    input: List[Node]

    def __init__(self, name: str, value: str, input: List[Node]) -> None:
        super().__init__(name, value)
        self.input = input

    def is_active(self) -> bool:
        # TDOD Threshold
        return max(map(self.input, lambda x: x.value)) == 1

    @abstractmethod
    def operation(self, data: List[float]):
        raise NotImplementedError()


class UnionNode(CombinedNode):
    def operation(self, data: float):
        return super().operation(data)


class IntersectionNode(CombinedNode):
    def operation(self, data: float):
        return super().operation(data)


class MajorityVoting(CombinedNode):
    def operation(self, data: float):
        return super().operation(data)


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
        # von hinten recursive aufrufen
        pass

    def valid(self) -> bool:
        last_level = self.levels[-1]
        one_is_active = False
        for i in [
            last_level.majority_voting,
            last_level.intersection,
            last_level.union,
        ]:
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

    def from_matrix(input_size: int, depth: int, matrix: List[List[float]]) -> Graph:
        g = Graph(depth, [InputNode("MD" + str(i), 0) for i in range(input_size)])


# %%

g = Graph(3, [InputNode("MD1", 0)])

# %%
g.to_dataframe()

# %%
