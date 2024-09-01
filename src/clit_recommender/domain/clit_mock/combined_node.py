from clit_recommender.domain.data_row import DataRow


from abc import ABC, abstractmethod
from typing import List

from clit_recommender.util import flat_map
from clit_recommender.domain.clit_mock.node import Node


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
