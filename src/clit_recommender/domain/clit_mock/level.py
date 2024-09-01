from clit_recommender.domain.clit_mock.combined_node import (
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)


from copy import deepcopy
from typing import List

from clit_recommender.domain.clit_mock.node import Node


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

    def OutputNodes(self) -> List[Node]:
        return deepcopy(self.input) + [
            self.majority_voting,
            self.intersection,
            self.union,
        ]
