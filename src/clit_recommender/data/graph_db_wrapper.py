# %%

from collections import defaultdict
from typing import Dict, List, Tuple
from SPARQLWrapper import JSON, SPARQLWrapper

from clit_recommender.domain.datasets import Dataset, DatasetSplitType
from clit_recommender.domain.systems import System


ACTUAL_KEY = "ACTUAL"


class GraphDBWrapper:

    _client: SPARQLWrapper
    _datasets: List[str]
    _systems: List[str]

    def __init__(
        self,
        datasets: List[Dataset] = list(Dataset),
        systems: List[System] = list(System),
        dataset_type: DatasetSplitType = DatasetSplitType.ALL,
    ) -> None:
        self._client = SPARQLWrapper(
            "http://localhost:7200/repositories/KDD", returnFormat=JSON
        )

        self._datasets = list(map(lambda s: f"<{dataset_type.get_uri(s)}>", datasets))
        self._systems = list(map(lambda s: f"<{s.uri}>", systems))

    def _query(self, query: str):
        query = f"""
            prefix aifb: <http://aifb.kit.edu/clit/recommender/>
            prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>

            {query}
            """
        self._client.setQuery(query)
        return map(
            lambda x: {key: value["value"] for key, value in x.items()},
            self._client.queryAndConvert()["results"]["bindings"],
        )

    def get_all_systems(self) -> List[str]:
        query = """
            select * 
            where { 
	            ?s a aifb:ClitMdSystem .
            } 
        """
        return [i["s"] for i in self._query(query)]

    def get_systems_on_datasets(self) -> List[str]:
        query = f"""
            select distinct ?s
            where {{ 
	            ?s a aifb:ClitMdSystem .
                ?r aifb:ofSystem ?s .
                ?r nif:referenceContext ?c .
                ?collection nif:hasContext ?c .
                values ?collection {{ {" ".join(self._datasets)} }} .
                
            }}
        """
        return [i["s"] for i in self._query(query)]

    def get_contexts(self, limit=999999, offset=0) -> List[Tuple[str, str]]:
        query = f"""
            select * 
            where {{
                ?c a nif:Context .
                ?c nif:isString ?t .
                ?collection nif:hasContext ?c .
                values ?collection {{ {" ".join(self._datasets)} }} .
            }}
            limit {limit}
            offset {offset}
        """
        return [(i["c"], i["t"]) for i in self._query(query)]

    def get_count(self) -> int:
        query = f"""
            select (count(*) as ?count)
            where {{
                ?c a nif:Context .
                ?c nif:isString ?t .
                ?collection nif:hasContext ?c .
                values ?collection {{ {" ".join(self._datasets)} }} .
            }}"""
        return int(next(self._query(query), None)["count"])

    def get_mentions_of_context(
        self, context_uri: str
    ) -> Dict[str, List[Tuple[str, int]]]:
        query = f"""
            select ?s ?t ?o 
            where {{
                
                ?m nif:referenceContext <{context_uri}> .
                ?m nif:beginIndex ?o .
                ?m nif:anchorOf ?t .
                OPTIONAL {{
                    ?m a aifb:ClitResult .
                    ?m aifb:ofSystem ?s .
                    values ?s {{ {" ".join(self._systems)} }} .
                }}
            }}
        """
        d = defaultdict(list)

        for i in self._query(query):
            m = (i["t"], int(i["o"]))
            s = i["s"] if "s" in i else ACTUAL_KEY
            d[s].append(m)
        return d


if __name__ == "__main__":
    g = GraphDBWrapper([Dataset.MED_MENTIONS])
    print(len(g.get_all_systems()))
    print(len(g.get_systems_on_datasets()))
    # print(g.get_count())
    # print(g.get_contexts())
    # print(g.get_mentions_of_context("http://med-mentions.niladi.de/all#char=0,100"))

# %%
