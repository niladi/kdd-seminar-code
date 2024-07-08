from clit_recommender.config import Config
from clit_recommender.models.base import ClitRecommenderModel


def model_factory(config: Config) -> ClitRecommenderModel:
    return config.model(config)
