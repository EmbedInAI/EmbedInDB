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

from sentence_transformers import SentenceTransformer

from embedin.embedding.embedding_base import EmbeddingBase


class SentenceTransformerEmbedding(EmbeddingBase):
    """
    A class for generating text embeddings using the SentenceTransformer model.

    Args:
    model_name (str): The name or path of the SentenceTransformer model to be used for generating embeddings. Default is "all-MiniLM-L6-v2".

    Methods:
    __call__(texts):
    Generates embeddings for the given input text(s).

    Returns:
    embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """
        Initialize a SentenceTransformerEmbedding object.

        Args:
        model_name (str): The name or path of the SentenceTransformer model to be used for generating embeddings. Default is "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model)

    def __call__(self, texts):
        """
        Generates embeddings for the given input text(s).

        Args:
        texts (str/list): The input text(s) for which embeddings are to be generated. It can be a string or a list of strings.

        Returns:
        embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
        """
        # Return it as a numpy array
        # Check if texts is a string
        if isinstance(texts, str):
            return self.model.encode(
                [texts], convert_to_numpy=True, show_progress_bar=True
            ).tolist()
        # Check if texts is a list of strings
        if isinstance(texts, list):
            if all(isinstance(text, str) for text in texts):
                return self.model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=True
                ).tolist()
            raise TypeError("Input must be a string, a list of strings")
        raise TypeError("Input must be a string, a list of strings")
