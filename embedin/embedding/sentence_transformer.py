from sentence_transformers import SentenceTransformer

from embedin.embedding import Embedding


class SentenceTransformerEmbedding(Embedding):
    """
    A class for generating text embeddings using the SentenceTransformer model.

    Args:
    model_name (str): The name or path of the SentenceTransformer model to be used for generating embeddings. Default is "all-MiniLM-L6-v2".

    Methods:
    __call__(texts):
    Generates embeddings for the given input text(s).

    Returns:
    embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize a SentenceTransformerEmbedding object.

        Args:
        model_name (str): The name or path of the SentenceTransformer model to be used for generating embeddings. Default is "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        """
        Generates embeddings for the given input text(s).

        Args:
        texts (str/list): The input text(s) for which embeddings are to be generated. It can be a string or a list of strings.

        Returns:
        embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
        """
        # Return it as a numpy array
        # Check if texts is a string
        if isinstance(texts, str):
            return self.model.encode([texts], convert_to_numpy=True).tolist()
        # Check if texts is a list of strings
        elif all(isinstance(text, str) for text in texts):
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        else:
            raise TypeError("Input must be a string, a list of strings")
