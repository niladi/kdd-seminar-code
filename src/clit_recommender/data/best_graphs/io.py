import json
from os import mkdir, remove
from typing import Dict, List
from os.path import join, exists

from clit_recommender import GraphPresentation
from clit_recommender.config import BEST_GRAPHS_JSON_FILE, BEST_GRAPHS_LMDB_FILE
from clit_recommender.data.best_graphs.base import BestGraphBase
from clit_recommender.data.lmdb_wrapper import LmdbImmutableDict


class BestGraphIO(BestGraphBase):

    def load_dict(self) -> Dict[str, List[GraphPresentation]]:
        best_graph = {}
        with open(join(self.path, BEST_GRAPHS_JSON_FILE), "r") as f:
            best_graph = json.load(f)
        return {
            key: list(
                tuple(tuple(float(value) for value in level) for level in graph)
                for graph in graphs
            )
            for key, graphs in best_graph.items()
        }

    def exists(self):
        return (
            exists(self.path)
            and exists(join(self.path, BEST_GRAPHS_JSON_FILE))
            and exists(join(self.path, BEST_GRAPHS_LMDB_FILE))
        )

    def load_lmdb(self):
        return LmdbImmutableDict(join(self.path, BEST_GRAPHS_LMDB_FILE))

    def save(self, best_graphs: Dict[str, List[GraphPresentation]]) -> None:
        if not exists(self.path):
            mkdir(self.path)

        with open(join(self.path, "config.json"), "w") as f:
            f.write(self.config.to_json())

        # Dump _best_graph to JSON
        with open(join(self.path, BEST_GRAPHS_JSON_FILE), "w") as f:
            json.dump(best_graphs, f)

        lmdb_path = join(self.path, BEST_GRAPHS_LMDB_FILE)

        if exists(lmdb_path):
            remove(lmdb_path)

        LmdbImmutableDict.from_dict(best_graphs, lmdb_path)
