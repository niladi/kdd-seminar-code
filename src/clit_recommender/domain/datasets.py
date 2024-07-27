from enum import Enum


@dataclass
class Dataset:
    name: str
    context_uri_prefix: str


class DatasetEnum(Dataset, Enum):
    MED_MENTIONS = ("MedMentions", "http://med-mentions.niladi.de/all")
    AIDA_YAGO2 = ("AidaConllYago", "https://aifb.kit.edu/conll")
    KORE_50 = (
        "Kore50",
        "http://www.mpi-inf.mpg.de/yago-naga/aida/download/KORE50.tar.gz/AIDA.tsv",
    )
    NEWS_100 = ("News100", "http://aksw.org/N3/News-100")
    REUTERS_128 = ("Reuters128", "http://aksw.org/N3/Reuters-128")
    RSS_500 = ("Rss500", "http://aksw.org/N3/RSS-500")
