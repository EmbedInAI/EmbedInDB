import unittest
from unittest.mock import Mock
from embedin.repository.collection_repository import CollectionRepository
from embedin.service.collection_service import CollectionService


class TestCollectionService(unittest.TestCase):

    def setUp(self):
        self.session = Mock()
        self.collection_repo = CollectionRepository(self.session)
        self.service = CollectionService(self.session)

    def test_get_by_name(self):
        # Define mock data
        name = 'test_collection'
        expected_rows = [{'name': name, 'id': 1}]
        # Mock dependency methods
        self.service.collection_repo.get_by_name = Mock(return_value=expected_rows)

        # Call the function being tested
        actual_rows = self.service.get_by_name(name)

        # Check the result
        self.assertEqual(actual_rows, expected_rows)
        self.service.collection_repo.get_by_name.assert_called_once_with(name)

    def test_create(self):
        # Define mock data
        name = 'test_collection'
        get_if_exist = True
        # Mock dependency methods
        self.service.collection_repo.create = Mock()

        # Call the function being tested
        self.service.create(name, get_if_exist)

        # Check the result
        self.service.collection_repo.create.assert_called_once_with(name, get_if_exist)
