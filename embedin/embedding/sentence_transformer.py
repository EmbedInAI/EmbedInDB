from sentence_transformers import SentenceTransformer

from embedin.embedding import Embedding


class SentenceTransformerEmbedding(Embedding):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        # Return it as a numpy array
        # Check if texts is a string
        if isinstance(texts, str):
            return self.model.encode([texts], convert_to_numpy=True).tolist()
        # Check if texts is a list of strings
        elif all(isinstance(text, str) for text in texts):
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        else:
            raise TypeError("Input must be a string, a list of strings")
