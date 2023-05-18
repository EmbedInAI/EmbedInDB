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

from abc import ABC, abstractmethod

from embedin.util import to_np_array


class IndexBase(ABC):
    def __init__(self, embeddings):
        """
        Args:
            embeddings: A list of embeddings, where each embedding is a list
            or array of floats.
        """
        self.embeddings = to_np_array(embeddings)
        self.index = self._build_index()

    @abstractmethod
    def _build_index(self):
        pass

    @abstractmethod
    def _search_index(self, query_embeddings, top_k):
        pass

    @abstractmethod
    def update_index(self, embeddings):
        pass

    def search(self, query_embeddings, top_k=3):
        """
        Perform nearest neighbor on a set of embeddings.

        Args:
            query_embeddings (list or array): The query embedding to use for the
            nearest neighbor search.
            top_k (int, optional): The number of nearest neighbors to return.
            Defaults to 3.

        Returns:
           A list of the indices of the nearest neighbors, and a list of their
           corresponding distances.
        """
        count = len(self.embeddings)
        top_k = min(top_k, count)
        query_embeddings = to_np_array(query_embeddings)
        indices = self._search_index(query_embeddings, top_k)
        return indices

    def get_embeddings(self):
        return self.embeddings
