# %%

from dataclasses import dataclass, field
from typing import Any, Dict, List

from dataclasses_json import dataclass_json
from numpy import mean
import numpy as np


@dataclass_json
@dataclass
class Metrics:

    num_gold_spans: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    num_docs: int = 0
    example_errors: List[Any] = field(default_factory=list)
    example_errors_md: List[Any] = field(default_factory=list)

    def __add__(self, other: "Metrics"):
        return Metrics(
            num_gold_spans=self.num_gold_spans + other.num_gold_spans,
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            num_docs=self.num_docs + other.num_docs,
            example_errors=self.example_errors + other.example_errors,
            example_errors_md=self.example_errors_md + other.example_errors_md,
        )

    def get_summary(self):
        p = self.get_precision()
        r = self.get_recall()
        f1 = self.get_f1()
        accuracy = self.get_accuracy()
        result = (
            f"\n****************\n"
            f"************\n"
            f"f1: {f1:.4f}\naccuracy: {accuracy:.4f}\np: {p:.4f}\nr: "
            f"{r:.4f}\nnum_gold_spans: {self.num_gold_spans}\n"
            f"************\n"
        )

        return result

    def get_precision(self):
        return self.tp / (self.tp + self.fp + 1e-8 * 1.0)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + 1e-8 * 1.0)

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2.0 * p * r / (p + r + 1e-8)

    def get_accuracy(self):
        return 1.0 * self.tp / (self.num_gold_spans + 1e-8)

    @classmethod
    def zeros(cls):
        return Metrics(num_gold_spans=0, tp=0, fp=0, fn=0)

    @classmethod
    def evaluate_matrices(cls, predicted, expected):

        real = np.matrix(predicted)
        predicted = np.matrix(expected)

        diff = real - predicted

        # Correct is 0
        # FP is -1
        # FN is 1

        tp = len(np.where(diff == 0)[0])
        fp = len(np.where(diff == -1)[0])
        fn = len(np.where(diff == 1)[0])

        metrics = Metrics(
            num_gold_spans=len(real),
            tp=tp,
            fp=fp,
            fn=fn,
            num_docs=1,
            example_errors=[],
        )
        return metrics

    @classmethod
    def evaluate_results(cls, gold, pred_spans, doc_title="", soft=True):
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
                    if pred_span[0] == gold_span[0]:
                        if (
                            pred_span[1] == gold_span[1]
                            or pred_span[1] - 1 == gold_span[1]
                            or pred_span[1] + 1 == gold_span[1]
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


@dataclass
class Epoch:
    """
    Represents an epoch in a training process.

    Attributes:
        metrics (Dict[str, Metrics]): A dictionary of evaluation metrics for entity linking (datset name, Metrics).
        loss (List[float]): A list of loss values for each iteration in the epoch.
    """

    result_metrics: Dict[str, Metrics] = field(default_factory=(lambda: {}))
    prediction_metrics: Dict[str, Metrics] = field(default_factory=(lambda: {}))
    loss: List[float] = field(default_factory=list)

    def get_loss_mean(self) -> float:
        """
        Calculates the mean loss value for the epoch.

        Returns:
            float: The mean loss value.
        """
        return mean(self.loss)

    def get_loss_total(self) -> float:
        """
        Calculates the total loss value for the epoch.

        Returns:
            float: The total loss value.
        """
        return sum(self.loss)

    def get_single_result(self) -> Metrics:
        """
        Returns the entity linking result if it is single

        Returns:
            Metrics: The el result
        """
        if len(self.result_metrics) == 1:
            return list(self.result_metrics.values())[0]
        else:
            raise ValueError("There are multiple results")


@dataclass_json
@dataclass
class MetricsHolder:
    epochs: List[Epoch] = field(default_factory=list)
    best_epoch_index: int = 0

    def add_epoch(self, epoch: Epoch = None) -> None:
        """
        Adds an epoch to the metrics holder.

        Parameters:
        - epoch: The epoch to be added. Defaults to a new instance of the Epoch class.

        Returns:
        - None
        """
        if epoch is None:
            epoch = Epoch()
        self.epochs.append(epoch)

    def add_loss_to_last_epoch(self, loss: float) -> None:
        """
        Adds the given loss value to the list of losses in the last epoch.

        Args:
            loss (float): The loss value to be added.

        Returns:
            None
        """
        self.get_last_epoch().loss.append(loss)

    def set_result_metrics_to_last_epoch(
        self, metric: Metrics, key: str = "Default"
    ) -> None:
        """
        Sets the given entity linking metric for the specified key in the el_metrics dictionary of the last epoch.

        Args:
            metric (Metrics): The metric to be set.
            key (str, optional): The key to associate with the metric (the dataset name of the metric for example). Defaults to "Default".

        Returns:
            None
        """
        self.get_last_epoch().result_metrics[key] = metric

    def set_prediction_metrics_to_last_epoch(
        self, metric: Metrics, key: str = "Default"
    ) -> None:
        """
        Sets the given entity linking metric for the specified key in the el_metrics dictionary of the last epoch.

        Args:
            metric (Metrics): The metric to be set.
            key (str, optional): The key to associate with the metric (the dataset name of the metric for example). Defaults to "Default".

        Returns:
            None
        """
        self.get_last_epoch().prediction_metrics[key] = metric

    def get_last_epoch(self) -> Epoch:
        """
        Returns the last (current) epoch in the metrics holder.

        Returns:
            The last epoch (current) in the metrics holder.
        """
        return self.epochs[-1]

    def get_best_epoch(self) -> Epoch:
        """
        Returns the best epoch based on the best_epoch_index. The best epoch is setted manually. (In ReFinED by the best el f1-score)

        Returns:
            The best epoch object.
        """
        return self.epochs[self.best_epoch_index]

    def set_last_epoch_as_best(self) -> None:
        """
        Sets the last (current) epoch as the best epoch.

        This method updates the `best_epoch_index` attribute to the index of the last (current) epoch in the `epochs` list.
        """
        self.best_epoch_index = len(self.epochs) - 1

    def get_mean_loss(self) -> List[float]:
        """
        get the mean loss value for each epoch.


        Returns:
            List[float]: mean losses value for each epoch.
        """
        return [epoch.get_loss_mean() for epoch in self.epochs]

    def get_total_loss(self) -> List[float]:
        """
        Get the total loss value for each epoch.

        Returns:
            List[float]: total losses value for each epoch.
        """
        return [epoch.get_loss_total() for epoch in self.epochs]

    def get_f1(self) -> List[float]:
        """
        Get the F1 value for each epoch.

        Returns:
            List[float]: F1 value for each epoch.
        """
        return [
            mean([m.get_f1() for m in epoch.result_metrics.values()])
            for epoch in self.epochs
        ]
