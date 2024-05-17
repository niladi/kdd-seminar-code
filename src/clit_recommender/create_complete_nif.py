import asyncio
from os import listdir, makedirs
from os.path import isdir, exists

from urllib.parse import quote

from pynif import NIFCollection, NIFContext
from rdflib import RDF, Graph, Literal, Namespace
from tqdm.auto import tqdm

from clit_recommender import DATASETS_PATH, CLIT_RESULTS_PATH, MD_ONLY
from clit_recommender.clit_result import ClitResult, Document, ExperimentTask, Mention
from clit_recommender.util import flat_map, iterate_dirs


_files_len = len(
    list(
        flat_map(
            lambda x: iterate_dirs(x, False),
            flat_map(iterate_dirs, iterate_dirs(MD_ONLY)),
        )
    )
)

_pbar = tqdm(total=_files_len)


def dataset_process(dataset: str, override: bool = False):
    aifb_namespace = Namespace("http://aifb.kit.edu/clit/recommender/")
    nif_namespace = Namespace(
        "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#"
    )
    dataset_path = f"{DATASETS_PATH}/{dataset}"
    nif_collection: NIFCollection
    try:
        nif_collection = NIFCollection.load(dataset_path, format="ttl")
    except Exception:
        print("Can't Parse dataset", dataset)
        return
    for system in listdir(f"{MD_ONLY}/{dataset}"):
        if not isdir(f"{MD_ONLY}/{dataset}/{system}"):
            continue

        if (not override) and exists(f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl"):
            print(
                f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl already exists",
                "Skipping",
            )
            continue

        system_uri = aifb_namespace[quote(system)]
        g_system = Graph()
        g_system.bind("aifb", aifb_namespace)
        g_system.bind("nif", nif_namespace)
        g_system.add((system_uri, RDF.type, aifb_namespace["ClitMdSystem"]))
        for experiment in listdir(f"{MD_ONLY}/{dataset}/{system}"):
            clit_result: ClitResult
            try:
                with open(f"{MD_ONLY}/{dataset}/{system}/{experiment}", "r") as f:
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
            _pbar.update(1)
        if not exists(f"{CLIT_RESULTS_PATH}/{dataset}"):
            makedirs(f"{CLIT_RESULTS_PATH}/{dataset}")
        g_system.serialize(
            f"{CLIT_RESULTS_PATH}/{dataset}/{system}.nif.ttl", format="ttl"
        )
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


for _dataset in listdir(DATASETS_PATH):
    print("Start", _dataset)
    dataset_process(_dataset)

_pbar.close()
