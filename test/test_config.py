import unittest

from clit_recommender.config import Config


class TestClitMock(unittest.TestCase):

    def test_json(self):
        cfg = Config()
        json = cfg.to_json()
        cfg_new = Config.from_json(json)

        self.assertEqual(cfg_new, cfg)
