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
from unittest.mock import patch

from embedin.embedding.openai_embedding import OpenAIEmbedding


class TestOpenAIEmbedding(unittest.TestCase):
    def setUp(self):
        self.api_key = "my_api_key"
        self.texts = ["hello", "world"]
        self.model = "text-davinci-002"
        self.embedding = OpenAIEmbedding(api_key=self.api_key, model=self.model)

    @patch("openai.Embedding.create")
    def test_call(self, mock_create):
        mock_create.return_value = {
            "data": [{"embedding": [1.0, 2.0, 3.0]}, {"embedding": [4.0, 5.0, 6.0]}]
        }
        result = self.embedding(self.texts)
        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_create.assert_called_once_with(model=self.model, input=self.texts)

    def tearDown(self):
        pass
