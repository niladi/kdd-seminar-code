import operator
from typing import Iterable
from clit_recommender.dataset import DataRow
from clit_recommender.inference import ClitRecommeder
from clit_recommender.metrics import Metrics
from tqdm.auto import tqdm


class Evaluation:
    processor: ClitRecommeder

    def __init__(self, processor) -> None:
        self.processor = processor

    def process_batch(self, batch: Iterable[DataRow]) -> Metrics:
        return sum(map(self.process_data_row, batch), Metrics.zeros())

    def process_data_row(self, data_row: DataRow) -> Metrics:
        gold = set(data_row.actual)
        predicted = self.processor.process_batch([data_row])

        pred_spans = set(
            filter(lambda x: x is not None, map(operator.itemgetter(0), predicted))
        )

        num_gold_spans = len(gold)
        tp = len(pred_spans & gold)
        fp = len(pred_spans - gold)
        fn = len(gold - pred_spans)

        fp_errors = sorted(list(pred_spans - gold), key=lambda x: x[1])[:5]
        fn_errors = sorted(list(gold - pred_spans), key=lambda x: x[1])[:5]

        metrics = Metrics(
            num_gold_spans=num_gold_spans,
            tp=tp,
            fp=fp,
            fn=fn,
            num_docs=1,
            example_errors=[
                {
                    "doc_title": data_row.context_text[:20],
                    "fp_errors": fp_errors,
                    "fn_errors": fn_errors,
                }
            ],
        )
        return metrics
