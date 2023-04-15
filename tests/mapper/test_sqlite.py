import unittest
from unittest.mock import patch

from src.mapper.sqlite import SQLite


class TestSQLite(unittest.TestCase):

    def setUp(self):
        self.sqlite = SQLite()
        cur = self.sqlite.conn.cursor()
        cur.execute("""DROP TABLE IF EXISTS collection;""")
        cur.execute("""DROP TABLE IF EXISTS embedding;""")
        cur.close()

    def tearDown(self):
        conn = self.sqlite.conn
        if conn:
            cur = conn.cursor()
            try:
                cur.execute("""DROP TABLE IF EXISTS collection;""")
                cur.execute("""DROP TABLE IF EXISTS embedding;""")
                conn.commit()
            except Exception as e:
                print(f"Error dropping tables: {e}")
            finally:
                cur.close()
            conn.close()

    def test_create_collection(self):
        with patch.object(self.sqlite, 'get_collection', return_value=[]):
            self.sqlite.create_collection('collection_name')
        self.assertEqual(len(self.sqlite.get_collection('collection_name')), 1)

    def test_create_collection_already_exists_with_exception(self):
        with patch.object(self.sqlite, 'get_collection', return_value=[(1, 'collection_name')]):
            self.sqlite.create_collection('collection_name')
        self.assertEqual(len(self.sqlite.get_collection('collection_name')), 0)

    def test_create_collection_already_exists_with_exception(self):
        with patch.object(self.sqlite, 'get_collection', return_value=[(1, 'collection_name')]):
            with self.assertRaises(ValueError):
                self.sqlite.create_collection('collection_name', get_if_exist=False)

    def test_add(self):
        collection_id = self.sqlite.create_collection('collection_name')[0][0]
        embeddings = [[1, 2, 3], [4, 5, 6]]
        texts = ['text1', 'text2']
        r1 = self.sqlite.add(collection_id, embeddings, texts)
        self.assertEqual(len(r1), 2)

        results = self.sqlite.get(collection_id=collection_id)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['embedding_data'], [1.0, 2.0, 3.0])
        self.assertEqual(results[1]['text'], 'text2')

        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        texts = ['text1', 'text2', 'text3']
        r2 = self.sqlite.add(collection_id, embeddings, texts)
        self.assertEqual(len(r2), 1)  # only 1 record is inserted

    def test_get_with_collection_name(self):
        collection_name = 'collection_name'
        collection_id = self.sqlite.create_collection(collection_name)[0][0]
        embeddings = [[1, 2, 3], [4, 5, 6]]
        texts = ['text1', 'text2']
        self.sqlite.add(collection_id, embeddings, texts)
        results = self.sqlite.get(collection_name=collection_name)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['embedding_data'], [1.0, 2.0, 3.0])
        self.assertEqual(results[1]['text'], 'text2')

    def test_get_with_invalid_collection_name(self):
        collection_name = 'collection_name'
        collection_id = self.sqlite.create_collection(collection_name)[0][0]
        embeddings = [[1, 2, 3], [4, 5, 6]]
        texts = ['text1', 'text2']
        self.sqlite.add(collection_id, embeddings, texts)
        with self.assertRaises(ValueError):
            self.sqlite.get(collection_name='invalid_collection')

    def test_close(self):
        self.sqlite.close()
        self.assertIsNone(self.sqlite.conn)
