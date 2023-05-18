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

import os

from embedin.embedding.openai_embedding import OpenAIEmbedding
from embedin.embedding.sentence_transformer import SentenceTransformerEmbedding


class Embedding:
    """
    A class for creating text embeddings using different models.

    Methods:
    create_embedding(model_type, api_key):
    Creates an instance of an embedding model based on the specified type and API key.

    __call__(texts):
    Raises a NotImplementedError. Subclasses must implement __call__() method.
    """

    @staticmethod
    def create_embedding(model_type, api_key=os.getenv("OPENAI_API_KEY")):
        """
        Creates an instance of an embedding model based on the specified type and API key.

        Args:
        model_type (str): The type of embedding model to create. Currently supported models are "openai" and "sentence_transformer".
        api_key (str): The API key required to use the OpenAI embedding model.

        Returns:
        An instance of the specified embedding model.
        """
        if model_type == "openai":
            if api_key is None:
                raise ValueError(
                    "Please set OPENAI_API_KEY environment variable. Windows: set OPENAI_API_KEY=your_api_key macOS/Linux: export OPENAI_API_KEY=your_api_key"
                )
            return OpenAIEmbedding(api_key=api_key)
        elif model_type == "sentence_transformer":
            return SentenceTransformerEmbedding()
        raise ValueError("Unsupported model type: {}".format(model_type))

    def __call__(self, texts):
        """
        Raises a NotImplementedError. Subclasses must implement __call__() method.

        Args:
        texts: The input text(s) for which embeddings are to be generated.

        Raises:
        NotImplementedError: This method is not implemented in the parent Embedding class.
        """
        raise NotImplementedError("Subclasses must implement __call__() method.")
