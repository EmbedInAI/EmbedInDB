import unittest

import numpy as np

from embedin.util import to_np_array


class TestToNpArray(unittest.TestCase):
    def test_to_np_array_none_input(self):
        query_embeddings = None
        with self.assertRaises(ValueError):
            to_np_array(query_embeddings)

    def test_to_np_array_empty_list(self):
        query_embeddings = []
        with self.assertRaises(ValueError):
            to_np_array(query_embeddings)

        query_embeddings = [[]]
        with self.assertRaises(ValueError):
            to_np_array(query_embeddings)

        query_embeddings = [[], []]
        with self.assertRaises(ValueError):
            to_np_array(query_embeddings)

    def test_to_np_array_one_dim(self):
        query_embeddings = [1, 2, 3]
        result = to_np_array(query_embeddings)
        assert result.shape == (1, 3)
        assert result.dtype == "float32"

    def test_to_np_array_two_dims(self):
        query_embeddings = [[1, 2, 3]]
        result = to_np_array(query_embeddings)
        assert result.shape == (1, 3)
        assert result.dtype == "float32"

    def test_to_np_array_two_dims_multiple(self):
        query_embeddings = [[1, 2, 3], [4, 5, 6]]
        result = to_np_array(query_embeddings)
        assert result.shape == (2, 3)
        assert result.dtype == "float32"

    def test_to_np_array_already_ndarray(self):
        query_embeddings = np.array([1, 2, 3])
        result = to_np_array(query_embeddings)
        assert result.shape == (1, 3)
        assert result.dtype == "float32"


if __name__ == "__main__":
    unittest.main()
