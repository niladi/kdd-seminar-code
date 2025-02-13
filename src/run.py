from multiprocessing import freeze_support
from clit_recommender.config import Config
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.systems import System
from clit_recommender.process.training import train_full

if __name__ == "__main__":
    freeze_support()
    _config = Config(epochs=20)
    #  _config.datasets = [Dataset.RSS_500]
    _config.systems = [
        System.BABEFLY,
        System.DBPEDIA_SPOTLIGHT,
        System.REFINED_MD_PROPERTIES,
        System.REL_MD_PROPERTIES,
        System.SPACY_MD_PROPERTIES,
        System.TAGME,
        System.TEXT_RAZOR,
    ]
    _config.fixed_size_combo = 3

    # _config.threshold = 0.3
    train_full(_config, True, True)
