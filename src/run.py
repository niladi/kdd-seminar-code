from multiprocessing import freeze_support
from clit_recommender.config import Config
from clit_recommender.domain.datasets import Dataset
from clit_recommender.domain.systems import System
from clit_recommender.process.training import train_full
import sys

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

    _config.margin_range = 0.05

    #float(str(margin_arg))
    # python
    #sys.argv[0])
    # fixed size
    fixed_size = 2
    if len(sys.argv) > 1:
        fixed_size = int(sys.argv[1])
    # margin
    margin = 0.05
    if len(sys.argv) > 2:
        margin = float(sys.argv[2])

    #sys.argv[2])

    _config.fixed_size_combo = fixed_size
    _config.margin_range = margin

    print(f"Running for fixed_size:   {_config.fixed_size_combo}")
    print(f"Running for margin_range: {_config.margin_range}")
    # _config.threshold = 0.3
    train_full(_config, True, True)
