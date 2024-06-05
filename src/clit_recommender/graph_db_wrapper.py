# %%

from collections import defaultdict
from typing import Dict, List, Tuple
from SPARQLWrapper import JSON, SPARQLWrapper


class GraphDBWrapper:

    _client: SPARQLWrapper

    def __init__(self) -> None:
        self._client = SPARQLWrapper(
            "http://localhost:7200/repositories/KDD", returnFormat=JSON
        )

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

    def get_systems(self) -> List[str]:
        query = """
            select * 
            where { 
	            ?s a aifb:ClitMdSystem .
            } 
        """
        return [i["s"] for i in self._query(query)]

    def get_contexts(self, limit=99999, offset=0) -> List[Tuple[str, str]]:
        query = f"""
            select * 
            where {{
                ?c a nif:Context .
                ?c nif:isString ?t .
            }}
            limit {limit}
            offset {offset}
        """
        return [(i["c"], i["t"]) for i in self._query(query)]

    def get_mentions_of_context(
        self, context_uri: str
    ) -> Dict[str, List[Tuple[str, int]]]:
        query = f"""
            select ?s ?t ?o 
            where {{
                ?m a aifb:ClitResult .
                ?m aifb:ofSystem ?s .
                ?m nif:referenceContext <{context_uri}> .
                ?m nif:beginIndex ?o .
                ?m nif:anchorOf ?t .
            }}
        """
        d = defaultdict(list)

        for i in self._query(query):
            d[i["s"]].append((i["t"], int(i["o"])))
        return d


# %%

w = GraphDBWrapper()

w.get_mentions_of_context(
    "http://www.mpi-inf.mpg.de/yago-naga/aida/download/KORE50.tar.gz/AIDA.tsv/CEL01#char=0,"
)

# %%
