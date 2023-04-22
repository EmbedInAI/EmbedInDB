import unittest
from datetime import datetime
from unittest.mock import Mock

from embedin.model.embedding_model import EmbeddingModel
from embedin.repository.collection_repository import CollectionRepository
from embedin.repository.embedding_repository import EmbeddingRepository
from embedin.service.embedding_service import EmbeddingService


class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        self.session = Mock()
        self.embedding_repo = EmbeddingRepository(self.session)
        self.collection_repo = CollectionRepository(self.session)
        self.service = EmbeddingService(self.session)

    def test_add_all(self):
        # Define mock data
        collection_id = 'test_collection'
        embeddings = [[1, 2, 3], [4, 5, 6]]
        texts = ['test_text_1', 'test_text_2']
        metadata_list = [{'meta1': 'value1'}, {'meta2': 'value2'}]
        expected_rows = [
            EmbeddingModel(
                id='test_id_1',
                collection_id=collection_id,
                text=texts[0],
                embedding_data=[1, 2, 3],
                meta_data=metadata_list[0],
                hash='test_hash_1',
                created_at=datetime.now()
            ),
            EmbeddingModel(
                id='test_id_2',
                collection_id=collection_id,
                text=texts[1],
                embedding_data=[4, 5, 6],
                meta_data=metadata_list[1],
                hash='test_hash_2',
                created_at=datetime.now()
            )
        ]
        # Mock dependency methods
        self.service.embedding_repo.add_all = Mock(return_value=expected_rows)

        # Call the function being tested
        actual_rows = self.service.add_all(collection_id, embeddings, texts, metadata_list)

        # Check the result
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(len(actual_rows), 2)
        self.assertEqual(actual_rows[0].hash, expected_rows[0].hash)
        self.assertEqual(actual_rows[0].embedding_data, embeddings[0])
        self.assertEqual(actual_rows[1].hash, expected_rows[1].hash)
        self.assertEqual(actual_rows[1].embedding_data, embeddings[1])

    def test_get_by_collection_id(self):
        # Define mock data
        collection_id = 'test_collection'
        embeddings = [[1, 2, 3], [4, 5, 6]]
        expected_rows = [
            EmbeddingModel(
                id='test_id_1',
                collection_id=collection_id,
                text='test_text_1',
                embedding_data=[1, 2, 3],
                meta_data=None,
                hash='test_hash_1',
                created_at=datetime.now()
            ),
            EmbeddingModel(
                id='test_id_2',
                collection_id=collection_id,
                text='test_text_2',
                embedding_data=[4, 5, 6],
                meta_data=None,
                hash='test_hash_2',
                created_at=datetime.now()
            )
        ]
        # Mock dependency methods
        self.service.embedding_repo.get_by_collection_id = Mock(return_value=expected_rows)

        # Call the function being tested
        actual_rows = self.service.get_by_collection_id(collection_id)

        # Check the result
        self.assertEqual(actual_rows, expected_rows)
        self.assertEqual(len(actual_rows), 2)
        self.assertEqual(actual_rows[0].hash, expected_rows[0].hash)
        self.assertEqual(actual_rows[0].embedding_data, embeddings[0])
        self.assertEqual(actual_rows[1].hash, expected_rows[1].hash)
        self.assertEqual(actual_rows[1].embedding_data, embeddings[1])
