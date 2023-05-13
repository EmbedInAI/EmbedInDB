import unittest
from unittest.mock import Mock, patch
from sqlalchemy import create_engine

from embedin.client import Client


class TestClient(unittest.TestCase):
    def setUp(self):
        self.collection_name = "test_collection"
        self.url = None
        self.embedding_fn = "sentence_transformer"
        self.index_hint = None
        self.debug = False

        self.mock_embedding = Mock()
        self.mock_embedding.create_embedding.return_value = self.mock_embedding

        self.mock_collection_service = Mock()
        self.mock_embedding_service = Mock()
        self.mock_index = Mock()

        self.mock_session = Mock()
        self.mock_session.return_value = self.mock_session
        self.mock_collection_service.get_by_name.return_value = None
        self.mock_embedding_service.get_by_collection_id.return_value = []

        self.mock_engine = create_engine("sqlite:///:memory:", echo=self.debug)

        self.patch_create_engine = patch("embedin.client.create_engine")
        self.mock_create_engine = self.patch_create_engine.start()
        self.mock_create_engine.return_value = self.mock_engine

        self.patch_Embedding = patch("embedin.embedding.Embedding")
        self.mock_Embedding = self.patch_Embedding.start()
        self.mock_Embedding.create_embedding.return_value = self.mock_embedding

        self.patch_CollectionService = patch("embedin.service.collection_service")
        self.mock_CollectionService = self.patch_CollectionService.start()
        self.mock_CollectionService.return_value = self.mock_collection_service

        self.patch_EmbeddingService = patch("embedin.service.embedding_service")
        self.mock_EmbeddingService = self.patch_EmbeddingService.start()
        self.mock_EmbeddingService.return_value = self.mock_embedding_service

        self.patch_Index = patch("embedin.index.Index")
        self.mock_Index = self.patch_Index.start()
        self.mock_Index.return_value = self.mock_index

        self.client = Client(
            self.collection_name,
            url=self.url,
            embedding_fn=self.embedding_fn,
            index_hint=self.index_hint,
            debug=self.debug,
        )

    def tearDown(self):
        self.patch_create_engine.stop()
        self.patch_Embedding.stop()
        self.patch_CollectionService.stop()
        self.patch_EmbeddingService.stop()
        self.patch_Index.stop()

    def test_init(self):
        self.mock_Embedding.create_embedding.assert_called_once_with(self.embedding_fn)

        # if self.url is None:
        #     self.mock_create_engine.assert_called_once_with(
        #         'sqlite:///:memory:', echo=self.debug)
        # else:
        #     self.mock_create_engine.assert_called_once_with(self.url)
        #
        # self.mock_session.assert_called_once_with(bind=self.mock_engine)
        # self.assertEqual(self.client.session, self.mock_session)

        # self.mock_CollectionService.assert_called_once_with(self.mock_session)
        # self.assertEqual(self.client.collection_service,
        #                  self.mock_collection_service)
        #
        # self.mock_EmbeddingService.assert_called_once_with(self.mock_session)
        # self.assertEqual(self.client.embedding_service,
        #                  self.mock_embedding_service)
        #
        # self.mock_collection_service.get_by_name.assert_called_once_with(
        #     self.collection_name)
        # self.mock_CollectionService.assert_called_once_with(self.mock_session)
        # self.mock_collection_service.create.assert_called_once_with(
        #     self.collection_name)
        # self.assertEqual(self.client.collection_id,
        #                  self.mock_collection_service.create.return_value)

        # self.mock_embedding_service.get_by_collection_id.assert_called_once_with(
        #     self.client.collection_id)
        # self.assertEqual(self.client.embedding_rows, [])
        #
        # self.assertEqual(self.client.index_hint, self.index_hint)
        # self.mock_Index.assert_called_once_with([], self.index_hint)
        # self.assertEqual(self.client.search_index, self.mock_index)

    def test_create_or_get_collection(self):
        self.mock
