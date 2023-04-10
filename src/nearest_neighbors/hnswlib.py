from hnswlib import Index as HNSWIndex
from src.nearest_neighbors import NearestNeighbors, to_np_array


class HNSWNearestNeighbors(NearestNeighbors):
    def _build_index(self):
        index = HNSWIndex('cosine', self.embeddings.shape[1])
        index.init_index(max_elements=self.embeddings.shape[0], ef_construction=100, M=16)
        index.add_items(self.embeddings)
        index.set_ef(100)
        return index

    def _search_index(self, query_embedding, k):
        query_embedding = to_np_array(query_embedding)
        indices, _ = self.index.knn_query(query_embedding, k=k)
        return indices[0]
