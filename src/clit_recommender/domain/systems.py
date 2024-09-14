from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class SystemEntry:
    index: int
    label: str
    uri: str


class System(SystemEntry, Enum):
    BABEFLY = (0, "Babelfy", "http://aifb.kit.edu/clit/recommender/Babelfy")
    CLOCQ = (1, "CLOCQ", "http://aifb.kit.edu/clit/recommender/CLOCQ")
    DBPEDIA_SPOTLIGHT = (
        2,
        "DBpediaSpotlight",
        "http://aifb.kit.edu/clit/recommender/DBpediaSpotlight",
    )
    FALCON_2 = (3, "Falcon 2.0", "http://aifb.kit.edu/clit/recommender/Falcon%202.0")
    OPEN_TAPIOCA = (
        4,
        "OpenTapioca",
        "http://aifb.kit.edu/clit/recommender/OpenTapioca",
    )
    REFINED_MD_PROPERTIES = (
        5,
        "Refined MD",
        "http://aifb.kit.edu/clit/recommender/Refined%20MD%20%28.properties%29",
    )
    REL = (6, "REL", "http://aifb.kit.edu/clit/recommender/REL")
    REL_MD_PROPERTIES = (
        7,
        "REL MD",
        "http://aifb.kit.edu/clit/recommender/REL%20MD%20%28.properties%29",
    )
    SPACY_MD_PROPERTIES = (
        8,
        "Spacy MD",
        "http://aifb.kit.edu/clit/recommender/Spacy%20MD%20%28.properties%29",
    )
    TAGME = (9, "TagMe", "http://aifb.kit.edu/clit/recommender/TagMe")
    TEXT_RAZOR = (10, "TextRazor", "http://aifb.kit.edu/clit/recommender/TextRazor")
