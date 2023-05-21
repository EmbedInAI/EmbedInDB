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

import datetime
import unittest
from unittest.mock import MagicMock

from embedin.model import Model


class ModelTestCase(unittest.TestCase):
    def test_to_dict(self):
        # Create a mock object for the self.__table__ attribute
        mock_table = MagicMock()

        # Mock the columns and their names
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col1.name = "col1"
        mock_col2.name = "col2"
        mock_table.c = [mock_col1, mock_col2]

        # Create a mock instance of the Model class
        model = Model()

        # Set mock values for the attributes
        model.col1 = "value1"
        model.col2 = datetime.datetime(2023, 5, 20)

        # Assign the mock_table directly to the __table__ attribute
        Model.__table__ = mock_table

        # Call the to_dict() method
        result = model.to_dict()

        # Check if the output is as expected
        expected = {"col1": "value1", "col2": "2023-05-20T00:00:00"}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
