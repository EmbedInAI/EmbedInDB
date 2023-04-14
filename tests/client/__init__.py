import os
import unittest
from unittest.mock import MagicMock
from src.client import Client


class TestClient(unittest.TestCase):
    def setUp(self):
        db_url = os.environ.get('DATABASE_URL',
                                "postgres://postgres:postgres@localhost:5432/test_db")
        # Initialize a test client
        self.client = Client(collection_name='test_collection',
                             url=db_url,
                             embedding_fn=MagicMock(return_value=[[1, 2, 3], [4, 5, 6]]))

    def test_create_collection(self):
        # Test that a collection is created with the given name
        collection_id = self.client.create_collection('new_collection')
        self.assertEqual(self.client.collection_id, collection_id)

    def test_add_data(self):
        # Test that data is added to the collection
        texts = ['text1', 'text2']
        meta_data_list = [{'meta1': 'value1'}, {'meta2': 'value2'}]
        self.client.add_data(texts, meta_data_list)
        self.assertEqual(len(self.client.embeddings), 2)

    def test_query(self):
        # Test that queries return the expected results
        self.client.add_data(['test', 'text'])
        result = self.client.query('test', top_k=1)
        expected_result = [{'text': 'test', 'metadata': None}]
        self.assertEqual(result, expected_result)

    def tearDown(self):
        # Clean up any resources created by the test

        conn = self.client.db.conn
        if conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                                DROP TABLE IF EXISTS collection, embedding CASCADE;
                            """)
                    conn.commit()
                except Exception as e:
                    print(f"Error dropping tables: {e}")
            conn.close()


if __name__ == '__main__':
    unittest.main()
