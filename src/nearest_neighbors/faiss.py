import faiss

from src.nearest_neighbors import NearestNeighbors


class FaissNearestNeighbors(NearestNeighbors):
    def _build_index(self):
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    def _search_index(self, query_embeddings, top_k):
        _, indices = self.index.search(query_embeddings, top_k)
        return indices[0]
