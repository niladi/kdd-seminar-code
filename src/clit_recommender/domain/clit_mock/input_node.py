from clit_recommender.domain.clit_mock.node import Node
from clit_recommender.domain.data_row import DataRow


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
