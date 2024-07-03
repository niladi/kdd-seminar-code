# %%

from collections import defaultdict
from typing import Dict, List, Tuple
from SPARQLWrapper import JSON, SPARQLWrapper

ACTUAL_KEY = "ACTUAL"


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

    def get_count(self) -> int:
        query = """
            select (count(*) as ?count)
            where {
                ?c a nif:Context .
                ?c nif:isString ?t .
            }"""
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
                }}
            }}
        """
        d = defaultdict(list)

        for i in self._query(query):
            m = (i["t"], int(i["o"]))
            s = i["s"] if "s" in i else ACTUAL_KEY
            d[s].append(m)
        return d
