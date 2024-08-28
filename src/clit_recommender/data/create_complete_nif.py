from abc import ABC, abstractmethod
from os import listdir, makedirs
from os.path import exists, isdir
from typing import List
from urllib.parse import quote

from pynif import NIFCollection, NIFContext
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from tqdm.auto import tqdm

from clit_recommender import CLIT_RESULTS_PATH, DATA_PATH, DATASETS_PATH, MD_ONLY
from clit_recommender.domain.clit_result import (
    ClitResult,
    Document,
    ExperimentTask,
    Mention,
)
from clit_recommender.domain.datasets import Dataset
from clit_recommender.util import flat_map, iterate_dirs


class NifFactory(ABC):
    _md_path: str
    _dataset_list: List[Dataset]
    _override: bool = False
    _pbar: tqdm

    def __init__(self, dataset_list: List[str], override: bool = False) -> None:

        self._dataset_list = dataset_list
        self._override = override

    @abstractmethod
    def _run(self, dataset: Dataset) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_progress_len(self) -> int:
        raise NotImplementedError()

    def _after_all(self) -> None:
        self._pbar.close()  # TODO Fix Sollte sauber funktionieren

    def __call__(self) -> None:
        return self.run()

    def run(self) -> None:
        self._pbar = tqdm(total=self._get_progress_len())
        for _dataset in self._dataset_list:
            print("Start", _dataset)
            self._run(_dataset)
        self._after_all()

    def get_nif_collection(self, path: str) -> NIFCollection:
        nif_collection: NIFCollection
        try:
            nif_collection = NIFCollection.load(path, format="ttl")
        except Exception:
            print("Can't Parse dataset", path)
            raise Exception("Invalid Dataset")
        return nif_collection


class NifAddCollectionUriFactory(NifFactory):
    _graph: Graph
    _nif_namespace: Namespace

    def __init__(self, dataset_list: List[str], override: bool = False) -> None:
        super().__init__(dataset_list, override)
        self._graph = Graph()

        self._nif_namespace = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        self._graph.bind("nif", self._nif_namespace)

    def _get_progress_len(self) -> int:
        return len(self._dataset_list)

    def _after_all(self) -> None:
        super()._after_all()
        file = f"{DATA_PATH}/dataset_context_extention.nif.ttl"

        if exists(file):
            print(file, "already exists")
            if not self._override:
                print("Skipping")
                return
        print("Start Saving ", file)
        self._graph.serialize(file, format="ttl")
        print("Done Saving")

    def _run(self, dataset: Dataset) -> None:
        dataset_path = f"{DATASETS_PATH}/{dataset.filename}"
        collection = self.get_nif_collection(dataset_path)
        collection_uri = URIRef(dataset.uri)

        self._graph.add(
            (
                collection_uri,
                RDF.type,
                self._nif_namespace["ContextCollection"],
            )
        )

        for context in collection.contexts:
            self._graph.add(
                (
                    collection_uri,
                    self._nif_namespace["hasContext"],
                    context.uri,
                )
            )

        self._pbar.update(1)


class NifClitResultFactory(NifFactory):
    def __init__(
        self, dataset_list: List[str], override: bool = False, md_path: str = MD_ONLY
    ) -> None:
        super().__init__(dataset_list, override)
        self._md_path = md_path

    def _get_progress_len(self) -> int:
        return len(
            list(
                flat_map(
                    lambda x: iterate_dirs(x, False),
                    flat_map(iterate_dirs, iterate_dirs(self._md_path)),
                )
            )
        )

    def _run(self, dataset: Dataset) -> None:
        dataset = dataset.filename
        aifb_namespace = Namespace("http://aifb.kit.edu/clit/recommender/")
        nif_namespace = Namespace(
            "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
        )
        dataset_path = f"{DATASETS_PATH}/{dataset}"
        nif_collection = self.get_nif_collection(dataset_path)

        for system in listdir(f"{self._md_path}/{dataset}"):
            if not isdir(f"{self._md_path}/{dataset}/{system}"):
                continue

            if (not self._override) and exists(
                f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl"
            ):
                print(
                    f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl already exists",
                    "Skipping",
                )
                self.__pbar.update(len(listdir(f"{self._md_path}/{dataset}/{system}")))
                continue

            system_uri = aifb_namespace[quote(system)]
            g_system = Graph(store="Oxigraph")
            g_system.bind("aifb", aifb_namespace)
            g_system.bind("nif", nif_namespace)
            g_system.add((system_uri, RDF.type, aifb_namespace["ClitMdSystem"]))
            for experiment in listdir(f"{self._md_path}/{dataset}/{system}"):
                clit_result: ClitResult
                try:
                    with open(
                        f"{self._md_path}/{dataset}/{system}/{experiment}", "r"
                    ) as f:
                        j = f.read()
                        clit_result = ClitResult.from_json(j)
                except Exception:
                    print("Can't Parse experiment", experiment)
                    continue
                task: ExperimentTask
                for task in clit_result.experiment_tasks:
                    document: Document
                    try:
                        documents = flat_map(lambda x: x, task.documents)
                    except:
                        print(type(task.documents))
                        print(task.documents)
                        raise Exception("To fix")
                    for document in documents:
                        context: NIFContext
                        found_context: NIFContext = None
                        for context in nif_collection.contexts:
                            if document.text == context.mention:
                                found_context = context
                                break
                        if found_context is None:
                            # RSS Fix
                            for context in nif_collection.contexts:
                                if document.text == context.mention.replace("\\", ""):
                                    found_context = context
                                    break

                        if found_context is None:
                            # MedMentions Fix

                            for context in nif_collection.contexts:
                                if (
                                    document.text.replace(".", "").strip()
                                    == context.mention.replace(".", "").strip()
                                ):
                                    found_context = context
                                    break

                        if found_context is None:
                            print(
                                "There should be at least one context of the the corresponding Query",
                                document.text,
                            )
                            continue

                        # query = f"""
                        # SELECT ?context ?phrase ?phrase_text ?phrase_begin_index
                        # WHERE {{
                        # ?context a nif:Context .
                        # ?context nif:isString ?text .
                        # FILTER (STR(?text)=""\"{text.replace('"','\\"')}\""") .
                        # OPTIONAL {{
                        # ?phrase a nif:Phrase .
                        # ?phrase nif:referenceContext ?context .
                        # ?phrase nif:anchorOf ?phrase_text .
                        # ?phrase nif:beginIndex ?phrase_begin_index
                        # }}
                        # }}
                        # """
                        # try:
                        # res = list(g.query(query))
                        # except Exception as e:
                        # print("Query", query)
                        # raise AssertionError("Query failed")
                        # if len(res) <= 0:
                        # print("Query", query)

                        mention: Mention
                        for mention in document.mentions:
                            mention_uri = aifb_namespace[
                                f"{quote(system)}/{quote(experiment)}?mention={quote(mention.mention)}&offset={mention.offset}"
                            ]
                            g_system.add(
                                (mention_uri, RDF.type, aifb_namespace["ClitResult"])
                            )
                            g_system.add(
                                (mention_uri, aifb_namespace["ofSystem"], system_uri)
                            )
                            g_system.add(
                                (
                                    mention_uri,
                                    nif_namespace["referenceContext"],
                                    found_context.uri,
                                )
                            )
                            g_system.add(
                                (
                                    mention_uri,
                                    nif_namespace["beginIndex"],
                                    Literal(mention.offset),
                                )
                            )
                            g_system.add(
                                (
                                    mention_uri,
                                    nif_namespace["anchorOf"],
                                    Literal(mention.mention),
                                )
                            )
                            # Für die verschiedenen Mentions die passenden NIF-Phrasen finden und mit den Mentions verknüpfen
                            # zu klären ob nur exact match oder auch partial match
                            # found_mention: Mention = None
                            # for res in res:
                            # if (
                            # mention.begin_index == res[3]
                            # and mention.text == res[2]
                            # ):
                            # found_mention = mention
                            # break
                self._pbar.update(1)
            if not exists(f"{CLIT_RESULTS_PATH}/{dataset}"):
                makedirs(f"{CLIT_RESULTS_PATH}/{dataset}")
            file = f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl"
            print("Start Saving ", file)
            g_system.serialize(file, format="ttl")
            print("Done Saving")
        print("Finished", dataset)


#
# TODO Fix asnyc rdflib bug
#
# async def create_complete_nif():
# async with asyncio.TaskGroup() as tg:
# for dataset in listdir(DATASETS_PATH):
# tg.create_task(dataset_process(dataset))
# print("Finished")
# pbar.close()


# asyncio.run(create_complete_nif())

if __name__ == "__main__":

    # Clit Results All Casual Domain
    # NifClitResultFactory([DatasetEnum.AIDA_YAGO2, DatasetEnum.KORE_50, DatasetEnum.NEWS_100, DatasetEnum.REUTERS_128, DatasetEnum.RSS_500], False, MD_ONLY)()

    # Clit Results Med Mentions Domain
    # NifClitResultFactory([DatasetEnum.MED_MENTIONS], False, MEDMENTION_PATH)()

    # Collection URL Casual Domain
    NifAddCollectionUriFactory(
        [Dataset.AIDA_YAGO2, Dataset.KORE_50],
        True,
    )()
