import unittest
from unittest.mock import MagicMock

import numpy as np

from embedin.client import Client


class TestClient(unittest.TestCase):
    def test_constructor(self):
        np.random.seed(42)
        embeddings = np.random.rand(3, 4).astype(np.float32).tolist()

        texts = ["hello", "test", "world"]

        client = Client(
            collection_name="test_collection", texts=texts, embeddings=embeddings
        )
        self.assertIsNotNone(client.collection_service)
        self.assertIsNotNone(client.embedding_service)

        client = Client(collection_name="test_collection", texts=texts)
        self.assertIsNotNone(client.collection_service)
        self.assertIsNotNone(client.embedding_service)

    def test_create_or_get_collection(self):
        collection_name = "test_collection"
        client = Client(collection_name=collection_name)
        collection_id = client.create_or_get_collection(collection_name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(client.collection_id, collection_id)

    def test_create_collection(self):
        # Test that a collection is created with the given name
        collection_name = "new_collection"
        client = Client(collection_name=collection_name)
        collection_id = client.create_collection(collection_name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(client.collection_id, collection_id)

    def test_get_collection(self):
        # Test that a collection is retrieved with the given name
        collection_name = "new_collection"
        client = Client(collection_name=collection_name)
        client.create_collection(collection_name)
        collection_id = client.get_collection(collection_name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(client.collection_id, collection_id)

    def test_add_data(self):
        collection_name = "new_collection"
        client = Client(collection_name=collection_name)

        # Test that data is added to the collection
        texts = ["text1", "text2"]
        meta_data = [{"meta1": "value1"}, {"meta2": "value2"}]

        client.embedding_fn = MagicMock(return_value=[[1, 2, 3], [4, 5, 6]])
        client.add_data(texts, meta_data)

        client.embedding_fn.assert_called_once_with(texts)
        self.assertEqual(len(client.embedding_rows), 2)

    def test_query(self):
        collection_name = "new_collection"
        client = Client(collection_name=collection_name)

        # Test that queries return the expected results
        client.embedding_fn = MagicMock(return_value=[[1, 2, 3], [4, 5, 6]])
        client.add_data(["test", "text"])

        client.embedding_fn = MagicMock(return_value=[[1, 2, 3]])
        result = client.query("test", top_k=1)
        expected_result = [{"text": "test", "meta_data": None}]
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
