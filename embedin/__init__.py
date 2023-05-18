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

"""
embedin - A vector database that empowers AI with persistent memory.

    >>> from embedin import Embedin
    >>> client = Embedin(collection_name="test_collection",
    ...                  texts=["This is a test", "Hello world!"])
    >>> result = client.query("These are tests", top_k=1)
    >>> print(result)

:copyright: (C) 2023 EmbedInAI
:license: Apache 2.0, see LICENSE for more details.
"""

from .client import Client as Embedin
