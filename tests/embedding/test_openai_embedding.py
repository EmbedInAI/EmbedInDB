import unittest
from unittest.mock import patch

from embedin.embedding.openai_embedding import OpenAIEmbedding


class TestOpenAIEmbedding(unittest.TestCase):
    def setUp(self):
        self.api_key = "my_api_key"
        self.texts = ["hello", "world"]
        self.model = "text-davinci-002"
        self.embedding = OpenAIEmbedding(api_key=self.api_key, model=self.model)

    @patch("openai.Embedding.create")
    def test_call(self, mock_create):
        mock_create.return_value = {
            "data": [{"embedding": [1.0, 2.0, 3.0]}, {"embedding": [4.0, 5.0, 6.0]}]
        }
        result = self.embedding(self.texts)
        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_create.assert_called_once_with(model=self.model, input=self.texts)

    def tearDown(self):
        pass
