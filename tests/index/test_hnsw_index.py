import unittest

import numpy as np
import torch

from hnswlib import Index
from embedin.index.hnsw_index import HNSWIndex


class TestHNSWNearestNeighbors(unittest.TestCase):
    def setUp(self):
        d = 64  # dimension
        nb = 10**5  # database size
        nq = 10  # nb of queries
        np.random.seed(1234)  # make reproducible
        self.xb = np.random.random((nb, d)).astype("float32")
        self.xb[:, 0] += np.arange(nb) / 1000.0
        self.xq = np.random.random((nq, d)).astype("float32")
        self.xq[:, 0] += np.arange(nq) / 1000.0

        self.top_k = 10

        cos = torch.nn.CosineSimilarity()
        output = cos(torch.from_numpy(self.xq[0]), torch.from_numpy(self.xb))
        result = torch.topk(output.flatten(), self.top_k).indices.numpy()
        self.gold_answer = np.array(
            [207, 381, 1394, 1019, 555, 210, 417, 1619, 837, 557]
        )
        self.assertTrue(np.array_equal(self.gold_answer, result))

        self.nn = HNSWIndex(self.xb)

    def test_init(self):
        self.assertTrue(np.array_equal(self.nn.embeddings, np.array(self.xb)))
        self.assertIsInstance(self.nn.index, Index)
        self.assertEqual(self.nn.index.get_current_count(), len(self.xb))

    def test_build_index(self):
        index = self.nn._build_index()
        self.assertTrue(isinstance(index, Index))

    def test_search_index(self):
        indices = self.nn._search_index(self.xq, top_k=10)
        self.assertTrue(np.array_equal(indices[0], self.gold_answer))

    def test_search(self):
        indices = self.nn.search(self.xq, top_k=10)
        self.assertTrue(np.array_equal(indices[0], self.gold_answer))

    def test_update_index(self):
        count = len(self.xb)
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = self.xq[0].tolist()
        self.nn.update_index(new_embeddings)
        count += 1
        self.assertEqual(self.nn.index.get_current_count(), count)
        indices = self.nn.search(self.xq, top_k=10)
        self.assertTrue(count - 1 in indices[0].tolist())

        new_embeddings = [self.xq[0].tolist(), self.xq[0].tolist()]
        self.nn.update_index(new_embeddings)
        count += len(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)
        indices = self.nn.search(self.xq, top_k=10)
        for i in range(len(new_embeddings)):
            self.assertTrue(count - 1 - i in indices[0].tolist())

        new_embeddings = []
        self.nn.update_index(new_embeddings)
        count += len(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)

        new_embeddings = None
        self.nn.update_index(new_embeddings)
        self.assertEqual(self.nn.index.get_current_count(), count)

    def test_get_embeddings(self):
        embeddings = self.nn.get_embeddings()
        self.assertTrue(np.array_equal(embeddings, np.array(self.xb)))
