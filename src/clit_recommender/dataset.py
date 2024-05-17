from os import listdir
from typing import Iterator
from pynif import NIFCollection
from torch.utils.data import IterableDataset

from clit_recommender import DATASETS_PATH


class ClitRecommenderDataset(IterableDataset):
    def __iter__(self) -> Iterator:
        for dataset in listdir(DATASETS_PATH):
            dataset_path = f"{DATASETS_PATH}/{dataset}"
            nif_collection: NIFCollection
            try:
                nif_collection = NIFCollection.load(dataset_path, format="ttl")
            except Exception:
                print("Can't Parse dataset", dataset)
                return
            for context in nif_collection.contexts:
                yield context

    def __getitem__(self, index):
        for i, x in enumerate(self):
            if i == index:
                return x

        raise IndexError()
