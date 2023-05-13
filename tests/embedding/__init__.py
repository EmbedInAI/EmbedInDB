import unittest
from unittest.mock import MagicMock

from embedin.embedding import Embedding, OpenAIEmbedding, SentenceTransformerEmbedding


class TestEmbedding(unittest.TestCase):
    def test_create_openai_embedding(self):
        model_type = "openai"
        api_key = "my_secret_api_key"
        embedding = Embedding.create_embedding(model_type, api_key)
        self.assertIsInstance(embedding, OpenAIEmbedding)

    def test_create_openai_embedding_without_api_key(self):
        model_type = "openai"
        api_key = None
        with self.assertRaises(ValueError):
            Embedding.create_embedding(model_type, api_key)

    def test_create_sentence_transformer_embedding(self):
        model_type = "sentence_transformer"
        embedding = Embedding.create_embedding(model_type, None)
        self.assertIsInstance(embedding, SentenceTransformerEmbedding)

    def test_create_unsupported_model_type(self):
        model_type = "unsupported"
        with self.assertRaises(ValueError):
            Embedding.create_embedding(model_type, None)

    def test_call_method_not_implemented(self):
        embedding = Embedding()
        with self.assertRaises(NotImplementedError):
            embedding("some text")


if __name__ == "__main__":
    unittest.main()
