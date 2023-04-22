import unittest

import numpy as np

from hnswlib import Index as HNSWIndex
from embedin.index.hnswlib import HNSWNearestNeighbors


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
        self.assertEqual(self.nn.index.get_current_count(), len(self.embeddings))

    def test_search(self):
        query_embeddings = [1, 1, 1]
        indices = self.nn.search(query_embeddings, top_k=2)
        self.assertTrue(np.array_equal(indices, np.array([2, 1])))

    def test_update_index(self):
        count = len(self.embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = [2, 4, 6]
        self.nn.update_index(new_embeddings)
        count += 1
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = [[2, 4, 6], [1, 3, 5]]
        self.nn.update_index(new_embeddings)
        count += len(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = []
        self.nn.update_index(new_embeddings)
        count += len(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = None
        self.nn.update_index(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)
