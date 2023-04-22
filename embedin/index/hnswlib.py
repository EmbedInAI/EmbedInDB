import numpy as np
from hnswlib import Index as HNSWIndex
from embedin.index import NearestNeighbors, to_np_array


class HNSWNearestNeighbors(NearestNeighbors):
    """
    This class implements a nearest neighbors search with HNSW algorithm using cosine similarity metric.
    Inherits from the abstract class 'NearestNeighbors'.
    """

    # TODO: save index to DB; load index from DB
    def _build_index(self):
        """
        Build an index using the HNSW algorithm with the cosine similarity metric and returns it.

        Returns:
        -------
        index: HNSWIndex
            The index built using HNSW algorithm.
        """

        # TODO: double check all those magic numbers
        index = HNSWIndex("cosine", self.embeddings.shape[1])
        index.init_index(
            max_elements=self.embeddings.shape[0], ef_construction=100, M=16
        )
        index.add_items(self.embeddings)
        index.set_ef(100)
        return index

    def update_index(self, embeddings):
        """
        Updates the index with new embeddings.

        Parameters:
        ----------
        embeddings: numpy array
            Embeddings to be added to the index.
        """

        if not embeddings:
            return
        embeddings = to_np_array(embeddings)
        new_index_size = self.index.get_current_count() + embeddings.shape[0]
        self.index.resize_index(new_index_size)
        self.index.add_items(embeddings)
        self.embeddings = np.concatenate((self.embeddings, embeddings), axis=0)

    def _search_index(self, query_embeddings, top_k):
        """
        Searches the index for the top K nearest embeddings to the given query embeddings.

        Parameters:
        ----------
        query_embeddings: numpy array
            Query embeddings to search the nearest neighbors.
        top_k: int
            Number of nearest embeddings to return.

        Returns:
        -------
        indices: numpy array
            Array of indices representing the nearest embeddings to the given query embeddings.
        """

        indices, _ = self.index.knn_query(query_embeddings, k=top_k)
        return indices[0]
