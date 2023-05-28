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

from embedin.embedding.sentence_transformer import SentenceTransformerEmbedding


class TestSentenceTransformerEmbedding(unittest.TestCase):
    def test_embedding_single_text(self):
        embedding = SentenceTransformerEmbedding()
        text = "This is a test sentence."
        expected_output = embedding.model.encode([text], convert_to_numpy=True)
        self.assertTrue((embedding(text) == expected_output).all())

    def test_embedding_multiple_texts(self):
        embedding = SentenceTransformerEmbedding()
        texts = ["This is a test sentence.", "This is another test sentence."]
        expected_output = embedding.model.encode(texts, convert_to_numpy=True)
        self.assertTrue((embedding(texts) == expected_output).all())

    def test_embedding_empty_text(self):
        embedding = SentenceTransformerEmbedding()
        text = ""
        expected_output = embedding.model.encode([text], convert_to_numpy=True)
        self.assertTrue((embedding(text) == expected_output).all())

    def test_embedding_invalid_input(self):
        embedding = SentenceTransformerEmbedding()
        invalid_input = 123
        with self.assertRaises(TypeError):
            embedding(invalid_input)

        invalid_input = [123, "string"]
        with self.assertRaises(TypeError):
            embedding(invalid_input)


if __name__ == "__main__":
    unittest.main()
