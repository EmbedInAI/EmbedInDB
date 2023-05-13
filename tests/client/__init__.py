import unittest
from unittest.mock import MagicMock

from embedin.client import Client


class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = Client("test_collection")

    def test_create_or_get_collection(self):
        name = "test_collection"
        collection_id = self.client.create_or_get_collection(name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(self.client.collection_id, collection_id)

    def test_create_collection(self):
        # Test that a collection is created with the given name
        name = "new_collection"
        collection_id = self.client.create_collection(name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(self.client.collection_id, collection_id)

    def test_get_collection(self):
        # Test that a collection is retrieved with the given name
        collection_name = "new_collection"
        self.client.create_collection(collection_name)
        collection_id = self.client.get_collection(collection_name)
        self.assertIsNotNone(collection_id)
        self.assertEqual(self.client.collection_id, collection_id)

    def test_add_data(self):
        # Test that data is added to the collection
        texts = ["text1", "text2"]
        meta_data = [{"meta1": "value1"}, {"meta2": "value2"}]

        self.client.embedding_fn = MagicMock(return_value=[[1, 2, 3], [4, 5, 6]])
        self.client.add_data(texts, meta_data)

        self.client.embedding_fn.assert_called_once_with(texts)
        self.assertEqual(len(self.client.embedding_rows), 2)

    def test_query(self):
        # Test that queries return the expected results
        self.client.embedding_fn = MagicMock(return_value=[[1, 2, 3], [4, 5, 6]])
        self.client.add_data(["test", "text"])

        self.client.embedding_fn = MagicMock(return_value=[[1, 2, 3]])
        result = self.client.query("test", top_k=1)
        expected_result = [{"text": "test", "meta_data": None}]
        self.assertEqual(result, expected_result)

    def tearDown(self):
        # Clean up any resources created by the test
        pass


if __name__ == "__main__":
    unittest.main()
