import torch
import unittest


from clit_recommender.config import Config
from clit_recommender.domain.clit_mock.combined_node import (
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)
from clit_recommender.domain.clit_mock.graph import Graph
from clit_recommender.domain.clit_result import Mention

import json

from clit_recommender.domain.data_row import DataRow


class TestClitMock(unittest.TestCase):

    def test_create(self):
        c = Config()
        random_tensor = torch.randn(c.calculate_output_size(), dtype=torch.float32)
        random_tensor = random_tensor.reshape(int(c.calculate_output_size() / 3), 3)
        g = Graph.create(c, random_tensor)

        self.assertEqual(json.dumps(random_tensor.tolist()), json.dumps(g.to_matrix()))

    def test_create_by_label(self):
        g = Graph.create_1_dim(
            Config(depth=2, md_modules_count=2),
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [0.7, 0.8, 0.1, 0.2, 0.4],
            MajorityVoting,
        )

        expected = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.4],
        ]

        self.assertEqual(json.dumps(g.to_matrix()), json.dumps(expected))

    def test_operation(self):
        row = DataRow(
            "test_uri",
            "test_text",
            [
                [Mention("Mention 1", 0), Mention("Mention 2", 3)],
                [Mention("Mention 2", 3), Mention("Mention 3", 6)],
            ],
            [Mention("Mention 2", 3)],
        )

        g = Graph.create_1_dim(
            Config(depth=1, md_modules_count=2),
            [],
            [0.9, 0.8],
            IntersectionNode,
        )

        self.assertSetEqual(g.forward(row), set([Mention("Mention 2", 3)]))

        g = Graph.create_1_dim(
            Config(depth=1, md_modules_count=2),
            [],
            [0.9, 0.8],
            UnionNode,
        )

        self.assertSetEqual(
            set(g.forward(row)),
            set(
                [
                    Mention("Mention 1", 0),
                    Mention("Mention 2", 3),
                    Mention("Mention 3", 6),
                ]
            ),
        )

        g = Graph.create_1_dim(
            Config(depth=1, md_modules_count=2),
            [],
            [0.9, 0.8],
            MajorityVoting,
        )

        self.assertSetEqual(g.forward(row), set([Mention("Mention 2", 3)]))
