import unittest

from embedin.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    def test_call_method_not_implemented(self):
        embedding = Embedding()
        with self.assertRaises(NotImplementedError):
            embedding(['some', 'texts'])


if __name__ == '__main__':
    unittest.main()
