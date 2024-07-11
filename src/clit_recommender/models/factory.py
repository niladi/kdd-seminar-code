from typing import Type

from clit_recommender.config import Config

from clit_recommender.models.one_depth import ClitRecommenderModelOneDepth
from clit_recommender.models.full import ClitRecommenderModelFull
from clit_recommender.models.base import ClitRecommenderModel


def model_factory(config: Config) -> ClitRecommenderModel:
    model = next(
        filter(
            lambda classz: classz.__name__ == config.model,
            [ClitRecommenderModelFull, ClitRecommenderModelOneDepth],
        )
    )

    return model(config)
