import unittest

import numpy as np

from hnswlib import Index as HNSWIndex
from src.nearest_neighbors.hnswlib import HNSWNearestNeighbors


class TestHNSWNearestNeighbors(unittest.TestCase):
    def setUp(self):
        self.embeddings = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        self.nn = HNSWNearestNeighbors(self.embeddings)

    def test_init(self):
        self.assertTrue(np.array_equal(self.nn.embeddings, np.array(self.embeddings)))
        self.assertIsInstance(self.nn.index, HNSWIndex)
        self.assertEqual(self.nn.count, len(self.embeddings))

    def test_search(self):
        query_embeddings = [1, 1, 1]
        indices = self.nn.search(query_embeddings, top_k=2)
        self.assertTrue(np.array_equal(indices, np.array([2, 1])))
