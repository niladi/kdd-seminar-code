from dataclasses import dataclass, field
from enum import Enum
from typing import List

from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass(frozen=True)
class DatasetEntry:
    label: str
    uri: str
    filename: str


class Dataset(DatasetEntry, Enum):
    MED_MENTIONS = (
        "MedMentions",
        "http://med-mentions.niladi.de/all",
        "medmentions.ttl",
    )

    AIDA_YAGO2 = (
        "AidaConllYago",
        "https://aifb.kit.edu/conll",
        "AIDA-YAGO2-dataset.tsv_nif",
    )
    KORE_50 = (
        "Kore50",
        "http://www.mpi-inf.mpg.de/yago-naga/aida/download/KORE50.tar.gz/AIDA.tsv",
        "KORE_50_DBpedia.ttl",
    )
    NEWS_100 = ("News100", "http://aksw.org/N3/News-100", "News-100.ttl")
    REUTERS_128 = ("Reuters128", "http://aksw.org/N3/Reuters-128", "Reuters-128.ttl")
    RSS_500 = ("Rss500", "http://aksw.org/N3/RSS-500", "RSS-500.ttl")


class DatasetSplitType(Enum):
    TRAIN = "/train"
    EVAL = "/eval"
    ALL = ""

    def get_uri(self, dataset: Dataset) -> str:
        return dataset.uri + self.value


if __name__ == "__main__":

    for dataset in Dataset:
        print(dataset.label)
        print(dataset.uri)
        print(dataset.filename)
