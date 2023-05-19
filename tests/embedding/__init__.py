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

from embedin.embedding import Embedding, OpenAIEmbedding, SentenceTransformerEmbedding


class TestEmbedding(unittest.TestCase):
    def test_create_openai_embedding(self):
        model_type = "openai"
        api_key = "my_secret_api_key"
        embedding = Embedding.create_embedding(model_type, api_key)
        self.assertIsInstance(embedding, OpenAIEmbedding)

    def test_create_openai_embedding_without_api_key(self):
        model_type = "openai"
        api_key = None
        with self.assertRaises(ValueError):
            Embedding.create_embedding(model_type, api_key)

    def test_create_sentence_transformer_embedding(self):
        model_type = "sentence_transformer"
        embedding = Embedding.create_embedding(model_type, None)
        self.assertIsInstance(embedding, SentenceTransformerEmbedding)

    def test_create_unsupported_model_type(self):
        model_type = "unsupported"
        with self.assertRaises(ValueError):
            Embedding.create_embedding(model_type, None)

    def test_call_method_not_implemented(self):
        embedding = Embedding()
        with self.assertRaises(NotImplementedError):
            embedding("some text")


if __name__ == "__main__":
    unittest.main()
