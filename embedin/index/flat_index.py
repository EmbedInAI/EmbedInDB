import faiss
import numpy as np

from embedin.index.index_base import IndexBase
from embedin.util import to_np_array


class FlatIndex(IndexBase):
    """
    This class implements a nearest neighbors search using brute-force cosine similarity metric.
    Inherits from the abstract class 'NearestNeighbors'.
    """

    def _build_index(self):
        """
        Build an index using the brute-force with the cosine similarity metric and returns it.

        Returns:
        -------
        index: faiss.IndexFlatIP
            The index built using brute-force.
        """

        d = self.embeddings.shape[1]  # dimension
        index = faiss.IndexFlatIP(d)

        xb_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        xb_normalized = self.embeddings / xb_norm

        index.add(xb_normalized)
        return index

    def update_index(self, embeddings):
        """
        Updates the index with new embeddings.

        Parameters:
        ----------
        embeddings: A list of embeddings, where each embedding is a list
            or array of floats.
        """

        if not embeddings:
            return
        embeddings = to_np_array(embeddings)

        xb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        xb_normalized = embeddings / xb_norm

        self.index.add(xb_normalized)
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

        # faiss.normalize_L2(query_embeddings)
        xq_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        xq_normalized = query_embeddings / xq_norm

        distances, indices = self.index.search(xq_normalized, k=top_k)

        # Remove the singleton dimension
        indices = np.squeeze(indices)

        return indices
