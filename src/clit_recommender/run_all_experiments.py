from copy import deepcopy
from multiprocessing import freeze_support
from clit_recommender import DATA_PATH
from clit_recommender.config import Config

from time import time
from os.path import join, exists
from os import rmdir

from clit_recommender.domain.systems import System
from clit_recommender.process.single_system import SingleSystem
from clit_recommender.process.training import train_full, cross_train


from tqdm.auto import tqdm


def run_all_experiments(config: Config):

    print("########### Start Single System #############")

    for data in tqdm(config.datasets):
        SingleSystem([data], config.results_dir).run_all()
    SingleSystem(config.datasets, config.results_dir).run_all()

    print("########### Single System Done #############")

    print("########### Start Train Full #############")

    for data in tqdm(config.datasets):
        cfg = deepcopy(config)
        cfg.experiment_name += "_" + data.label
        cfg.datasets = [data]
        train_full(cfg, True)

    train_full(config, True)

    print("########### Train Full Done #############")

    print("########### Start Cross Train #############")

    for idx, data in enumerate(tqdm(config.datasets)):
        cfg = deepcopy(config)
        cfg.experiment_name += "_" + data.label
        training_sets = cfg.datasets[:idx] + cfg.datasets[idx + 1 :]
        cross_train(cfg, training_sets, [data], True)

    print("########### Cross Train Done #############")


if __name__ == "__main__":
    freeze_support()
    _config = Config()
    _config.systems = [
        System.BABEFLY,
        System.DBPEDIA_SPOTLIGHT,
        System.OPEN_TAPIOCA,
        System.REFINED_MD_PROPERTIES,
        System.REL_MD_PROPERTIES,
        System.SPACY_MD_PROPERTIES,
        System.TAGME,
        System.TEXT_RAZOR,
    ]
    _config.results_dir = join(DATA_PATH, "results_full")
    if exists(_config.results_dir):
        rmdir(_config.results_dir)
    run_all_experiments(_config)
