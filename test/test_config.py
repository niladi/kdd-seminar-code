from copy import deepcopy
import unittest

from clit_recommender.config import Config


class TestClitMock(unittest.TestCase):

    def test_json(self):
        cfg = Config()
        cfg.threshold = 10.5
        cfg.results_dir = "balabla"
        json = cfg.to_json()

        cfg_new = Config.from_json(json)

        self.assertDictEqual(cfg_new.to_dict(), cfg.to_dict())

    def test_deepcopy(self):
        config = Config()
        cfg = config
        cfg.datasets.pop(1)
        self.assertEqual(config, cfg)
        cfg = deepcopy(config)
        cfg.datasets.pop(1)
        self.assertNotEqual(config, cfg)
