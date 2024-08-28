from torch import Tensor, nn

from clit_recommender.config import Config
from clit_recommender.data.dataset.DataRowWithBestGraph import DataRowWithBestGraph
from clit_recommender.models.base import ClitRecommenderModel, ModelResult
from clit_recommender.models.clit_mock import Graph


class ClitRecommenderModelOneDepth(ClitRecommenderModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._hidden_layer = nn.Linear(
            self._embedding_size, self._embedding_size, device=config.device
        )
        self._classification_layer = nn.Linear(
            self._embedding_size, 3, device=config.device
        )
        self._multi_label_layer = nn.Linear(
            self._embedding_size, config.md_modules_count, device=config.device
        )

    def forward(self, embeddings: Tensor, data_row: DataRowWithBestGraph):
        embeddings = embeddings.to(self._config.device).view(-1)
        hidden_output = self._hidden_layer(embeddings)
        classification_output: Tensor = self._classification_layer(hidden_output)

        multi_label_output = self._multi_label_layer(hidden_output)

        if data_row is not None:
            values, g_type = Graph.create(
                self._config, data_row.best_graph
            ).get_last_level_tuple()
            target = [0.0] * 3
            target[g_type.get_index()] = 1.0
            classification_loss = nn.CrossEntropyLoss()(
                classification_output, Tensor(target).to(device=self._config.device)
            )
            multi_label_loss = nn.BCEWithLogitsLoss()(
                multi_label_output, Tensor(values).to(device=self._config.device)
            )
            total_loss = classification_loss + multi_label_loss

        graph = Graph.create_1_dim(
            self._config,
            [],
            multi_label_output.sigmoid(),
            classification_output.softmax(dim=0).argmax().item(),
        )

        return ModelResult(graph.to_matrix(), total_loss)
