from torch import Tensor, nn

from clit_recommender.config import Config

from clit_recommender.data.dataset.clit_recommender_data_set import DataRowWithBestGraph
from clit_recommender.domain.model_result import ModelResult
from clit_recommender.models.base import ClitRecommenderModel
from clit_recommender.domain.clit_mock.graph import Graph


class ClitRecommenderModelOneDepth(ClitRecommenderModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        layers = []
        h_size = config.model_hidden_layer_size
        for i in range(config.model_depth):
            if i == 0:
                i_size = self._embedding_size
            else:
                i_size = h_size

            layers.append(nn.Linear(i_size, h_size, device=config.device))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

        self._hidden_layer = nn.Sequential(*layers)

        self._classification_layer = nn.Linear(h_size, 3, device=config.device)
        self._multi_label_layer = nn.Linear(
            h_size, config.md_modules_count, device=config.device
        )

    def forward(self, embeddings: Tensor, data_row: DataRowWithBestGraph):
        embeddings = embeddings.to(self._config.device).view(-1)
        hidden_output = self._hidden_layer(embeddings)
        classification_output: Tensor = self._classification_layer(hidden_output)

        multi_label_output = self._multi_label_layer(hidden_output)
        total_loss = None

        if data_row is not None:
            values, g_type = Graph.create(
                self._config, data_row.best_graph
            ).get_last_level_tuple_roundet()
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
