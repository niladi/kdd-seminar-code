import torch
import unittest

from clit_recommender.clit_mock import (
    Graph,
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)
from clit_recommender.config import Config
from clit_recommender.clit_result import Mention
from clit_recommender.dataset import DataRow


class TestClitMock(unittest.TestCase):
    def test_create(self):
        c = Config()
        random_tensor = torch.randn(c.calculate_output_size())
        random_tensor = random_tensor.reshape(int(c.calculate_output_size() / 3), 3)
        g = Graph.create(c, random_tensor)

        self.assertEqual(str(random_tensor.tolist()), str(g.to_matrix()))

    def test_create_by_label(self):
        g = Graph.create_by_last_as_vector_and_label(
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

        self.assertEqual(str(g.to_matrix()), str(expected))

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

        g = Graph.create_by_last_as_vector_and_label(
            Config(depth=1, md_modules_count=2),
            [],
            [0.9, 0.8],
            IntersectionNode,
        )

        self.assertSetEqual(g.forward(row), set([Mention("Mention 2", 3)]))

        g = Graph.create_by_last_as_vector_and_label(
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
