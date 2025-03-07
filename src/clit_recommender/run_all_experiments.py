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
from clit_recommender.domain.metrics import MetricType


from tqdm.auto import tqdm


def run_all_experiments(config: Config):

    print("########### Start Single System #############")

    SingleSystem(config.datasets, config.results_dir).run_all()
    for data in tqdm(config.datasets):
        SingleSystem([data], config.results_dir).run_all()

    print("########### Single System Done #############")

    print("########### Start Train Full #############")
    train_full(config, True, True)

    for data in tqdm(config.datasets):
        cfg = deepcopy(config)
        cfg.experiment_name += "_" + data.label
        cfg.datasets = [data]
        train_full(cfg, True)

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
    _config = Config(epochs=20)
    _config.systems = [
        System.BABEFLY,
        System.DBPEDIA_SPOTLIGHT,
        System.REFINED_MD_PROPERTIES,
        System.REL_MD_PROPERTIES,
        System.SPACY_MD_PROPERTIES,
        System.TAGME,
        System.TEXT_RAZOR,
    ]

    for metric_type in list(MetricType):
        _cfg = deepcopy(_config)
        _cfg.results_dir = join(DATA_PATH, "results_full", metric_type.value.lower())
        if exists(_cfg.results_dir):
            rmdir(_cfg.results_dir)
        _cfg.metric_type = metric_type
        _cfg.experiment_name = _cfg.create_name()
        run_all_experiments(_cfg)
