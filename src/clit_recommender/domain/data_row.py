from clit_recommender import GraphPresentation
from clit_recommender.domain.clit_result import Mention


from dataclasses import dataclass
from typing import List, Self


@dataclass
class DataRow:
    context_uri: str
    context_text: str
    results: List[List[Mention]]
    actual: List[Mention]

    def __hash__(self) -> int:
        return hash(self.context_uri)


@dataclass
class DataRowWithBestGraph(DataRow):
    best_graph: GraphPresentation

    @classmethod
    def from_data_row(cls, row: DataRow, best_graph: GraphPresentation) -> Self:
        return cls(
            row.context_uri, row.context_text, row.results, row.actual, best_graph
        )

    def __hash__(self) -> int:
        return hash((self.context_uri, self.best_graph))
