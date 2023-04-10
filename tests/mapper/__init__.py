import unittest

from src.mapper import DB


class TestDB(unittest.TestCase):
    def test_abstract_methods(self):
        with self.assertRaises(TypeError):
            DB()


if __name__ == '__main__':
    unittest.main()