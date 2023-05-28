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
        name = "test_collection"
        expected_rows = [{"name": name, "id": 1}]
        # Mock dependency methods
        self.service.collection_repo.get_by_name = Mock(return_value=expected_rows)

        # Call the function being tested
        actual_rows = self.service.get_by_name(name)

        # Check the result
        self.assertEqual(actual_rows, expected_rows)
        self.service.collection_repo.get_by_name.assert_called_once_with(name)

    def test_create(self):
        # Define mock data
        name = "test_collection"
        get_if_exist = True
        # Mock dependency methods
        self.service.collection_repo.create = Mock()

        # Call the function being tested
        self.service.create(name, get_if_exist)

        # Check the result
        self.service.collection_repo.create.assert_called_once_with(name, get_if_exist)
