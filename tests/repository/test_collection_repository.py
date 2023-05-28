# -*- coding: utf-8 -*-
# embedin - A vector database that empowers AI with persistent memory,
# (C) 2023 EmbedInAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import TestCase, mock
from unittest.mock import Mock, patch, MagicMock

from embedin.model.collection_model import CollectionModel
from embedin.repository.collection_repository import CollectionRepository


class TestCollectionRepository(TestCase):
    def setUp(self):
        self.session_mock = Mock()
        self.repo = CollectionRepository(self.session_mock)

    def test_get_by_name(self):
        # Mocking a CollectionModel object
        collection = CollectionModel(id="123", name="test_collection")
        self.session_mock.query.return_value.filter_by.return_value.first.return_value = (
            collection
        )

        # Call the method and assert the result
        result = self.repo.get_by_name("test_collection")
        self.assertEqual(result, collection.to_dict())

        # Verify that the query was executed with the correct arguments
        self.session_mock.query.assert_called_once_with(CollectionModel)
        self.session_mock.query.return_value.filter_by.assert_called_once_with(
            name="test_collection"
        )
        self.session_mock.query.return_value.filter_by.return_value.first.assert_called_once()

    def test_get_by_name_return_empty_dict(self):
        self.session_mock.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # Call the method and assert the result
        result = self.repo.get_by_name("test_collection")
        self.assertEqual(result, {})

    def test_create(self):
        # call create method
        name = "test_collection"

        # mock the get_by_name method
        with patch.object(self.repo, "get_by_name", return_value=None):
            collection = self.repo.create(name)
            self.assertIsInstance(collection, dict)
            self.assertIsNotNone(collection)
            self.assertEqual(collection.get("name"), name)
            self.assertIsInstance(collection.get("id"), str)

    def test_create_raise_exception(self):
        # call create method
        name = "test_collection"
        # Mocking a CollectionModel object
        mock_collection = CollectionModel(id="123", name=name)
        with patch.object(self.repo, "get_by_name", return_value=mock_collection):
            with self.assertRaises(ValueError):
                self.repo.create(name, get_if_exist=False)

    def test_create_with_commit_error(self):
        # Configure the commit method of the mock session to raise an exception
        self.session_mock.commit.side_effect = Exception("Mocked commit error")

        name = "test_collection"
        with patch.object(self.repo, "get_by_name", return_value=None):
            collection = self.repo.create(name, get_if_exist=True)
            self.assertIsNone(collection)
            self.session_mock.rollback.assert_called_once_with()

    def test_create_already_exists(self):
        # call create method
        name = "test_collection"

        # Mocking a CollectionModel object
        mock_collection = dict(id="123", name=name)
        with patch.object(self.repo, "get_by_name", return_value=mock_collection):
            collection = self.repo.create(name)
            self.assertIsInstance(collection, dict)
            self.assertIsNotNone(collection)
            self.assertEqual(collection.get("name"), mock_collection["name"])
            self.assertEqual(collection.get("id"), mock_collection["id"])
