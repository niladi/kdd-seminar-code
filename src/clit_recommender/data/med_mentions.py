from enum import Enum


from pynif import NIFCollection

from tqdm import tqdm

from clit_recommender import DATASETS_PATH, MEDMENTION_PATH
from clit_recommender.domain.datasets import Dataset


class MedMentionsType(Enum):
    ALL = "all"
    TRAIN = "trng"
    DEV = "dev"
    TEST = "test"

    def get_collection_uri(self):
        return f"https://pubmed.ncbi.nlm.nih.gov/"

    def get_entities_uri(self):
        return f"https://uts.nlm.nih.gov/uts/umls/concept/"

    def get_types_uri(self):
        return f"https://uts.nlm.nih.gov/uts/umls/semantic-network/"


#
# Orgininal Data can be found under https://github.com/chanzuckerberg/MedMentions
#
def parse_med_mentions_to_nif(split_type: MedMentionsType, result_file_name: str):
    valid_pmids = (
        open(f"{MEDMENTION_PATH}/corpus_pubtator_pmids_{split_type.value}.txt", "r")
        .read()
        .splitlines()
    )

    lines = open(f"{MEDMENTION_PATH}/corpus_pubtator.txt", "r").read().splitlines()

    collection_uri = split_type.get_collection_uri()

    collection = NIFCollection(uri=collection_uri)
    current_title = ""
    current_abstract = ""
    context = None
    skip_pm = False
    for line in tqdm(lines):
        if line.strip() == "":
            # End of a document
            current_title = ""
            current_abstract = ""
            continue

        parts = line.strip().split("|")
        if len(parts) == 3:
            if parts[0] not in valid_pmids:
                skip_pm = True
                continue
            else:
                skip_pm = False
            context = None
            if parts[1] == "t":
                current_title = parts[2]
            elif parts[1] == "a":
                current_abstract = parts[2]
        else:
            if skip_pm:
                continue

            mention_parts = line.strip().split("\t")

            if len(mention_parts) != 6:
                continue

            pmid, start, end, mention, semantic_types, entity_id = mention_parts

            if context is None:
                context_uri = f"{collection_uri}/{pmid}"
                context = collection.add_context(
                    uri=context_uri,
                    mention=f"{current_title} {current_abstract}",
                )
            # Process mention lines

            semantic_types_uri = [
                f"{split_type.get_types_uri()}#{s}" for s in semantic_types.split(",")
            ]

            start, end = int(start), int(end)

            context.add_phrase(
                uri=f"{context_uri}#{hash(mention)}_{start}_{end}",
                beginIndex=start,
                endIndex=end,
                is_hash_based_uri=True,
                taClassRef=semantic_types_uri,
                taIdentRef=f"{split_type.get_entities_uri()}#{entity_id}",
            )
    file_path = f"{DATASETS_PATH}/{result_file_name}"
    print(f"Saving: {file_path}")
    collection.dump(file_path)
    print("Done Saving!")


if __name__ == "__main__":
    parse_med_mentions_to_nif(MedMentionsType.ALL, Dataset.MED_MENTIONS.filename)
