from copy import deepcopy
import random
from typing import List

from rdflib import RDF, URIRef
from rdflib.namespace import Namespace
from rdflib.graph import Graph

from tqdm.auto import tqdm

from os.path import join

from clit_recommender import DATA_PATH
from clit_recommender.config import Config
from clit_recommender.data.dataset.clit_result_dataset import ClitResultDataset
from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.datasets import Dataset, DatasetSplitType


class CreateTrainEvalCollection:
    def __init__(self, config: Config) -> None:
        self._graph = Graph()

        self._nif_namespace = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        self._graph.bind("nif", self._nif_namespace)
        self._config = config
        self._pb = tqdm(total=len(ClitResultDataset(config)))

    def on_dataset(self, dataset: Dataset, seed: int = 500, eval_factor: int = 0.15):
        c = deepcopy(self._config)
        c.datasets = [dataset]
        data_list = list(ClitResultDataset(c))
        random.seed(seed)
        random.shuffle(data_list)

        eval_size = int(len(data_list) * eval_factor)

        eval = data_list[:eval_size]
        train = data_list[eval_size:]
        data: List[DataRow]

        for t, data in {
            DatasetSplitType.TRAIN: train,
            DatasetSplitType.EVAL: eval,
        }.items():
            collection_uri = URIRef(t.get_uri(dataset))
            self._graph.add(
                (
                    collection_uri,
                    RDF.type,
                    self._nif_namespace["ContextCollection"],
                )
            )
            for batch in data:
                for d in batch:
                    self._pb.update(1)
                    self._graph.add(
                        (
                            collection_uri,
                            self._nif_namespace["hasContext"],
                            URIRef(d.context_uri),
                        )
                    )

    def on_all_datasets(self, seed: int = 500, eval_factor: int = 0.15):
        for dataset in self._config.datasets:
            self.on_dataset(dataset, seed, eval_factor)

        f = join(DATA_PATH, "train_eval_collections.ttl")
        self._graph.serialize(f)


if __name__ == "__main__":

    CreateTrainEvalCollection(Config()).on_all_datasets()
