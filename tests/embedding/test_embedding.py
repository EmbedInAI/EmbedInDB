import unittest

from embedin.embedding.sentence_transformer import SentenceTransformerEmbedding


class TestSentenceTransformerEmbedding(unittest.TestCase):
    def test_embedding_single_text(self):
        embedding = SentenceTransformerEmbedding()
        text = "This is a test sentence."
        expected_output = embedding.model.encode([text], convert_to_numpy=True)
        self.assertTrue((embedding(text) == expected_output).all())

    def test_embedding_multiple_texts(self):
        embedding = SentenceTransformerEmbedding()
        texts = ["This is a test sentence.", "This is another test sentence."]
        expected_output = embedding.model.encode(texts, convert_to_numpy=True)
        self.assertTrue((embedding(texts) == expected_output).all())

    def test_embedding_empty_text(self):
        embedding = SentenceTransformerEmbedding()
        text = ""
        expected_output = embedding.model.encode([text], convert_to_numpy=True)
        self.assertTrue((embedding(text) == expected_output).all())

    def test_embedding_invalid_input(self):
        embedding = SentenceTransformerEmbedding()
        invalid_input = 123
        with self.assertRaises(TypeError):
            embedding(invalid_input)


if __name__ == "__main__":
    unittest.main()
