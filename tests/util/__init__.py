import unittest

from embedin.index import to_np_array


class TestToNpArray(unittest.TestCase):
    def test_to_np_array_zero_dim(self):
        query_embeddings = []
        with self.assertRaises(ValueError):
            result = to_np_array(query_embeddings)

    def test_to_np_array_one_dim(self):
        query_embeddings = [1, 2, 3]
        result = to_np_array(query_embeddings)
        assert result.shape == (1, 3)

    def test_to_np_array_two_dims(self):
        query_embeddings = [[1, 2, 3]]
        result = to_np_array(query_embeddings)
        assert result.shape == (1, 3)

    def test_to_np_array_two_dims_multiple(self):
        query_embeddings = [[1, 2, 3], [4, 5, 6]]
        result = to_np_array(query_embeddings)
        assert result.shape == (2, 3)


if __name__ == "__main__":
    unittest.main()
