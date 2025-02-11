from typing import List, Set, Tuple, Type, Union
from copy import deepcopy

import pandas as pd
import numpy as np

from torch import Tensor

from clit_recommender import GraphPresentation
from clit_recommender.domain.clit_mock.level import Level
from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.clit_mock.input_node import InputNode
from clit_recommender.domain.clit_mock.combined_node import (
    CombinedNode,
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)
from clit_recommender.config import Config
from clit_recommender.domain.clit_result import Mention


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
            input_node = self.levels[-1].OutputNodes()

    def forward(self, data_row: DataRow) -> Set[Mention]:
        level = self.get_last_level_node()
        return set() if level is None else set(level.calc(data_row))

    def valid(self) -> bool:
        last_level = self.levels[-1]
        one_is_active = False
        for i in last_level.OutputNodes():
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
        for i in last_level.OutputNodes():
            if i.is_active():
                return i

    def get_last_level_tuple_roundet(
        self,
    ) -> Tuple[Tuple[float, ...], Type[CombinedNode]]:
        node = self.get_last_level_node()
        if node is None:
            return None
        return tuple(
            map(lambda x: 1.0 if x.value >= self.threshold else 0.0, node.input)
        ), type(node)

    @staticmethod
    def create_1_dim(
        config: Config,
        # ignored for depth=1
        value_matrix: GraphPresentation,
        # this is what's done for final depth
        last_level_values: Union[List[float], Tensor, Tuple[float, ...]],
        # it's either a combine node or an int (0, 1, 2) aka. )
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
