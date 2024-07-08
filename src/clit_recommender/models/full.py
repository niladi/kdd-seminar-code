import operator
from data.dataset import DataRow

from torch import Tensor, nn

from clit_recommender.models.base import ClitRecommenderModel
from clit_recommender.config import Config
from models.clit_mock import Graph


class ClitRecommenderLoss(nn.Module):
    def __init__(self):
        super(ClitRecommenderLoss, self).__init__()

    def forward(self, outputs, targets):

        gold = set(targets)

        if outputs is None:
            return 1

        pred_spans = set(map(operator.itemgetter(1), outputs))

        num_gold_spans = len(gold)
        tp = len(pred_spans & gold)
        fp = len(pred_spans - gold)

        # Calculate precision
        precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero

        # We minimize (1 - precision) as the loss
        loss = 1 - precision
        return loss


class ClitRecommenderModelFull(ClitRecommenderModel):
    _loss: ClitRecommenderLoss

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Linear function
        self._fc1 = nn.Linear(
            config.lm_hidden_size, config.lm_hidden_size, device=config.device
        )
        # Non-linearity
        self._sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self._fc2 = nn.Linear(
            config.lm_hidden_size, config.calculate_output_size(), device=config.device
        )

        self._loss = ClitRecommenderLoss()

    def forward(self, embeddings: Tensor, data_row: DataRow):
        embeddings = embeddings.to(self._config.device)
        logits: Tensor = self._fc1(embeddings)
        logits = self._sigmoid(logits)
        logits = self._fc2(logits)

        logits = logits.reshape(int(self._config.calculate_output_size() / 3), 3)

        if data_row is not None:
            result = Graph.create(self._config, logits.tolist()).forward(data_row)
            loss = self._loss(result, data_row.actual)

        return result, logits, loss
