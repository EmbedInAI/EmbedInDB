from abc import ABC, abstractmethod

import numpy as np


def to_np_array(query_embedding):
    return np.array(query_embedding, dtype=np.float32).reshape(1, -1)


class NearestNeighbors(ABC):
    def __init__(self, embeddings):
        """
        Args:
            embeddings: A list of embeddings, where each embedding is a list
            or array of floats.
        """
        self.embeddings = np.array(embeddings).astype('float32')
        self.index = self._build_index()
        self.count = self.embeddings.shape[0]

    @abstractmethod
    def _build_index(self):
        pass

    @abstractmethod
    def _search_index(self, query_embedding, k):
        pass

    def search(self, query_embedding, k=3):
        """
        Perform nearest neighbor on a set of embeddings.

        Args:
            query_embedding (list or array): The query embedding to use for the
            nearest neighbor search.
            k (int, optional): The number of nearest neighbors to return.
            Defaults to 3.

        Returns:
           A list of the indices of the nearest neighbors, and a list of their
           corresponding distances.
        """
        k = min(k, self.count)
        indices = self._search_index(query_embedding, k)
        return indices
