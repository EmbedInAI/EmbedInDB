from embedin.index.flat_index import FlatIndex
from embedin.index.hnsw_index import HNSWIndex


class Index:
    def __init__(self, embeddings, index_hint=None):
        if index_hint == "hnsw":
            self.index = HNSWIndex(embeddings)
        elif index_hint == "flat":
            self.index = FlatIndex(embeddings)
        elif len(embeddings) > 10**6:
            self.index = HNSWIndex(embeddings)
        else:
            self.index = FlatIndex(embeddings)

    def search(self, query, top_k=3):
        return self.index.search(query, top_k)

    def update_index(self, embeddings):
        self.index.update_index(embeddings)
        # TODO: switch index when exceeding 10 ** 6
