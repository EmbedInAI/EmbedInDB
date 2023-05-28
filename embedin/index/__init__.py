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

from embedin.index.flat_index import FlatIndex
from embedin.index.hnsw_index import HNSWIndex


class Index:
    def __init__(self, embeddings, index_hint=None):
        if index_hint == "hnsw":
            self.index = HNSWIndex(embeddings)
        elif index_hint == "flat":
            self.index = FlatIndex(embeddings)
        elif len(embeddings) > 10**6:
            self.index = HNSWIndex(embeddings)
        else:
            self.index = FlatIndex(embeddings)

    def search(self, query, top_k=3):
        return self.index.search(query, top_k)

    def update_index(self, embeddings):
        self.index.update_index(embeddings)
        # TODO: switch index when exceeding 10 ** 6
