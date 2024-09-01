from typing import Iterable, Tuple

from clit_recommender.config import Config
from clit_recommender.domain.clit_mock.graph import Graph
from clit_recommender.domain.data_row import DataRow
from clit_recommender.process.inference import ClitRecommeder
from clit_recommender.domain.metrics import Metrics


class Evaluation:
    processor: ClitRecommeder

    def __init__(self, processor: ClitRecommeder) -> None:
        self.processor = processor

    def process_batch(self, batch: Iterable[DataRow]) -> Metrics:
        return sum(map(self.process_data_row, batch), Metrics.zeros())

    def process_data_row(self, data_row: DataRow) -> Tuple[Metrics, Metrics]:
        gold = set(data_row.actual)
        result = self.processor.process_batch(data_row)
        config = self.processor._config
        g = Graph.create(config, result.logits)
        predicted = g.forward(data_row)

        pred_spans = set(predicted)
        return (
            Evaluation.evaluate_results(gold, pred_spans, data_row.context_text[:20]),
            Evaluation.evaluate_matrices(
                g, Graph.create(config, data_row.best_graph), config.threshold
            ),
        )

    @staticmethod
    def evaluate_matrices(predicted: Graph, expected: Graph, threshold: float):

        tp = fp = fn = num = 0

        if len(expected.levels) > 1:
            for l in expected.levels[:-1]:
                for i_n, n in enumerate(l.input):
                    num += 1
                    if n.value == 1:
                        if predicted.levels[l.level].input[i_n].value >= threshold:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if predicted.levels[l.level].input[i_n].value >= threshold:
                            fp += 1

        expected_tuple = expected.get_last_level_tuple_roundet()
        predicted_tuple = predicted.get_last_level_tuple_roundet()

        if expected_tuple is None:
            return Metrics.zeros()  # TDOD Fix? case 0000.. is best graph

        last_exptected, last_exptected_type = expected_tuple
        if predicted_tuple is None:
            fp += 1  # todo: is this correct? The not Selected Type
            fn += sum(last_exptected)
        else:
            last_predicted, last_predicted_type = predicted_tuple
            if last_exptected_type == last_predicted_type:
                tp += 3  # todo: is this correct?
            else:
                fp += 1  # todo: is this correct?
                fn += 1  # todo: is this correct?
                tp += 1  # todo: is this correct?

            num += 3

            tp += sum(1 for a, b in zip(last_predicted, last_exptected) if a == b == 1)
            fp += sum(
                1 for a, b in zip(last_predicted, last_exptected) if a == 1 and b == 0
            )
            fn += sum(
                1 for a, b in zip(last_predicted, last_exptected) if a == 0 and b == 1
            )

        num += len(last_exptected)

        metrics = Metrics(
            num_gold_spans=num,
            tp=tp,
            fp=fp,
            fn=fn,
            num_docs=1,
            example_errors=[],
        )
        return metrics

    @staticmethod
    def evaluate_results(gold, pred_spans, doc_title="", soft=True):
        num_gold_spans = len(gold)
        if not soft:
            tp = pred_spans & gold
            fp = pred_spans - gold
            fn = gold - pred_spans
        else:
            tp, fp = set(), set()
            for pred_span in pred_spans:
                in_gold = False
                for gold_span in gold:
                    if pred_span[0] == gold_span[0]:  # Is text the Same
                        if (
                            pred_span[1] == gold_span[1]  # Offset is the same
                            or pred_span[1] - 1 == gold_span[1]  # Offset is -1
                            or pred_span[1] + 1 == gold_span[1]  # Offset is +1
                        ):
                            tp.add(gold_span)
                            in_gold = True
                            break
                if not in_gold:
                    fp.add(pred_span)
            fn = gold - tp

        fp_errors = sorted(fp, key=lambda x: x[1])[:5]
        fn_errors = sorted(fn, key=lambda x: x[1])[:5]

        metrics = Metrics(
            num_gold_spans=num_gold_spans,
            tp=len(tp),
            fp=len(fp),
            fn=len(fn),
            num_docs=1,
            example_errors=[
                {
                    "doc_title": doc_title,
                    "fp_errors": fp_errors,
                    "fn_errors": fn_errors,
                }
            ],
        )
        return metrics
