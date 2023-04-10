from hnswlib import Index as HNSWIndex
from src.nearest_neighbors import NearestNeighbors


class HNSWNearestNeighbors(NearestNeighbors):
    def _build_index(self):
        index = HNSWIndex('cosine', self.embeddings.shape[1])
        index.init_index(max_elements=self.embeddings.shape[0], ef_construction=100, M=16)
        index.add_items(self.embeddings)
        index.set_ef(100)
        return index

    def _search_index(self, query_embeddings, top_k):
        indices, _ = self.index.knn_query(query_embeddings, k=top_k)
        return indices[0]
