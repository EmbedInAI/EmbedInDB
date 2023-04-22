from abc import ABC, abstractmethod

from embedin.util import to_np_array


class NearestNeighbors(ABC):
    def __init__(self, embeddings):
        """
        Args:
            embeddings: A list of embeddings, where each embedding is a list
            or array of floats.
        """
        self.embeddings = to_np_array(embeddings)
        self.index = self._build_index()

    @abstractmethod
    def _build_index(self):
        pass

    @abstractmethod
    def _search_index(self, query_embeddings, top_k):
        pass

    @abstractmethod
    def update_index(self, embeddings):
        pass

    def search(self, query_embeddings, top_k=3):
        """
        Perform nearest neighbor on a set of embeddings.

        Args:
            query_embeddings (list or array): The query embedding to use for the
            nearest neighbor search.
            top_k (int, optional): The number of nearest neighbors to return.
            Defaults to 3.

        Returns:
           A list of the indices of the nearest neighbors, and a list of their
           corresponding distances.
        """
        count = len(self.embeddings)
        top_k = min(top_k, count)
        query_embeddings = to_np_array(query_embeddings)
        indices = self._search_index(query_embeddings, top_k)
        return indices
