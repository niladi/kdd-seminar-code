from config import Config
from data.dataset import DataRow
from torch import Tensor
from clit_recommender.models.base import ClitRecommenderModel


import torch.nn as nn


class ClitRecommenderModelOneDepth(ClitRecommenderModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._hidden_layer = nn.Linear(config.input_size, config.hidden_size)
        self._classification_layer = nn.Linear(config.hidden_size, 3)
        self._multi_label_layer = nn.Linear(config.hidden_size, config.md_modules_count)
        self._classification_loss = nn.CrossEntropyLoss()
        self._multi_label_loss = nn.BCEWithLogitsLoss()

    def forward(self, embeddings: Tensor, data_row: DataRow):
        hidden_output = self._hidden_layer(embeddings)
        classification_output = self._classification_layer(hidden_output)
        multi_label_output = self._multi_label_layer(hidden_output)

        classification_loss = self._classification_loss(
            classification_output, data_row.classification_target
        )
        multi_label_loss = self._multi_label_loss(
            multi_label_output, data_row.multi_label_target
        )
        total_loss = classification_loss + multi_label_loss

        return total_loss, classification_output, multi_label_output
