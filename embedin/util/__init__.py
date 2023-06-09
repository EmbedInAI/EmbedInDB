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

import numpy as np


def to_np_array(embeddings, dtype="float32"):
    """
    Convert a list of embeddings to a numpy array.

    Args:
        embeddings (list): A list of embeddings, where each element is a number or
        a list of numbers.
        dtype (string): Data type of the array

    Returns:
        numpy.ndarray: A 2D numpy array of shape (num_examples, embedding_size), where
        `num_examples` is the number of embeddings in the input list and `embedding_size`
        is the length of each individual embedding.

    Raises:
        ValueError: If the input list is empty or contains embeddings of inconsistent length.

    Example:
        >>> embeddings_list = [[1,2,3],[4,5,6]]
        >>> to_np_array(embeddings_list)
        array([[1, 2, 3],
               [4, 5, 6]], dtype=float32)
    """
    if embeddings is None:
        raise ValueError("Input list cannot be None.")

    if len(embeddings) == 0:
        raise ValueError("Input list cannot contain empty list")

    embeddings_array = np.array(embeddings, dtype=dtype)

    if embeddings_array.size == 0:
        raise ValueError("Input list cannot be empty")

    # If the array has only one dimension, add an extra dimension
    if len(embeddings_array.shape) == 1:
        embeddings_array = np.expand_dims(embeddings_array, axis=0)

    return embeddings_array
