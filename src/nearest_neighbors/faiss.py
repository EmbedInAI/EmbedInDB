import faiss

from src.nearest_neighbors import NearestNeighbors, to_np_array


class FaissNearestNeighbors(NearestNeighbors):
    def _build_index(self):
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    def _search_index(self, query_embedding, k):
        query_embedding = to_np_array(query_embedding)
        _, indices = self.index.search(query_embedding, k)
        return indices[0]
