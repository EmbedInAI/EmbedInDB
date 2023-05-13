import unittest

from embedin.embedding.embedding_base import EmbeddingBase


class TestEmbedding(unittest.TestCase):
    def test_call_method_not_implemented(self):
        embedding = EmbeddingBase()
        with self.assertRaises(NotImplementedError):
            embedding(["some", "texts"])
