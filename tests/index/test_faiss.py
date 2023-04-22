import unittest

import numpy as np

import faiss
from embedin.index.faiss import FaissNearestNeighbors


class TestFaissNearestNeighbors(unittest.TestCase):
    def setUp(self):
        self.embeddings = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        self.nn = FaissNearestNeighbors(self.embeddings)

    def test_init(self):
        self.assertTrue(np.array_equal(self.nn.embeddings, np.array(self.embeddings)))
        self.assertEqual(len(self.nn.embeddings), len(self.embeddings))
        self.assertIsInstance(self.nn.index, faiss.IndexFlatIP)

    def test_search(self):
        query_embeddings = [1, 1, 1]
        indices = self.nn.search(query_embeddings, top_k=2)
        self.assertTrue(np.array_equal(indices, np.array([2, 1])))
