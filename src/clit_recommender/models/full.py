import operator


from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.metrics import Metrics

from torch import Tensor, nn

from clit_recommender.domain.model_result import ModelResult
from clit_recommender.models.base import ClitRecommenderModel
from clit_recommender.config import Config
from clit_recommender.domain.clit_mock.graph import Graph


class ClitRecommenderLoss(nn.Module):
    def __init__(self):
        super(ClitRecommenderLoss, self).__init__()

    def forward(self, outputs, targets):

        metrics = Metrics.evaluate_results(outputs, targets)
        return 1 - metrics.get_f1()


# Multi Level/ Depth
class ClitRecommenderModelFull(ClitRecommenderModel):
    _loss: ClitRecommenderLoss

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Linear function
        self._fc1 = nn.Linear(
            self._embedding_size, self._embedding_size, device=config.device
        )
        # Non-linearity
        self._sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self._fc2 = nn.Linear(
            self._embedding_size, config.calculate_output_size(), device=config.device
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

        return ModelResult(logits, loss)
