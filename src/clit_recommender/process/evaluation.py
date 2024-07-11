import operator
from typing import Iterable, Tuple

from clit_recommender.models.clit_mock import Graph
from clit_recommender.data.dataset import DataRow
from clit_recommender.process.inference import ClitRecommeder
from clit_recommender.domain.metrics import Metrics
from tqdm.auto import tqdm


class Evaluation:
    processor: ClitRecommeder

    def __init__(self, processor) -> None:
        self.processor = processor

    def process_batch(self, batch: Iterable[DataRow]) -> Metrics:
        return sum(map(self.process_data_row, batch), Metrics.zeros())

    def process_data_row(self, data_row: DataRow) -> Tuple[Metrics, Metrics]:
        gold = set(data_row.actual)
        result = self.processor.process_batch([data_row])[0]
        predicted = Graph.create(self.processor._config, result.logits).forward(
            data_row
        )

        pred_spans = set(predicted)
        return Metrics.evaluate_results(gold, pred_spans, data_row.context_text[:20])
