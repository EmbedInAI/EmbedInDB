import unittest

import numpy as np

from embedin.index import Index, HNSWIndex, FlatIndex


class TestIndex(unittest.TestCase):
    def test_index_selection(self):
        # Test when index_hint is hnsw
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        index = Index(embeddings, "hnsw")
        self.assertIsInstance(index.index, HNSWIndex)

        # Test when index_hint is flat
        index = Index(embeddings, "flat")
        self.assertIsInstance(index.index, FlatIndex)

        # Test when embeddings length is less than or equal to 10^6
        embeddings = np.random.random((10**6 - 1, 3)).astype("float32")
        index = Index(embeddings)
        self.assertIsInstance(index.index, FlatIndex)

        # Test when embeddings length is greater than 10^6
        embeddings = np.random.random((10**6 + 1, 3)).astype("float32")
        index = Index(embeddings)
        self.assertIsInstance(index.index, HNSWIndex)

        # Test when embeddings is None
        embeddings = np.random.random((4, 3)).astype("float32")
        index = Index(embeddings)
        self.assertIsInstance(index.index, FlatIndex)

    def test_search(self):
        # Test search method with top_k=3
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        index = Index(embeddings)
        query = np.array([1, 2, 3], dtype="float32")
        results = index.search(query)
        self.assertEqual(len(results), 3)

        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        index = Index(embeddings)
        query = [1, 2, 3]
        results = index.search(query)
        self.assertEqual(len(results), 3)

    def test_update_index(self):
        # Test update_index method
        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        index = Index(embeddings)
        new_embeddings = [[10, 11, 12], [13, 14, 15]]
        index.update_index(new_embeddings)
        results = index.search([10, 11, 12])
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
