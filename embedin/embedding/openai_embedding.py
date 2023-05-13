import openai

from embedin.embedding.embedding_base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    """
    A class for generating text embeddings using the OpenAI API.

    Args:
    api_key (str): The API key for authenticating with the OpenAI API.
    model (str, optional): The name or ID of the OpenAI model to use for generating embeddings. Default is "text-embedding-ada-002".

    Methods:
    __call__(texts):
    Generates embeddings for the given input text(s).

    Returns:
    embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
    """

    def __init__(self, api_key, model="text-embedding-ada-002"):
        """
        Initialize an OpenAIEmbedding object.

        Args:
        api_key (str): The API key for authenticating with the OpenAI API.
        model (str, optional): The name or ID of the OpenAI model to use for generating embeddings. Default is "text-embedding-ada-002".
        """
        openai.api_key = api_key
        self.model = model

    def __call__(self, texts):
        """
        Generates embeddings for the given input text(s) using the OpenAI API.

        Args:
        texts (str/list): The input text(s) for which embeddings are to be generated. It can be a string or a list of strings.

        Returns:
        embeddings (list): A list of embeddings generated for the input text(s). Each embedding is a list of float values.
        """
        embeddings = openai.Embedding.create(model=self.model, input=texts)
        embeddings = [data["embedding"] for data in embeddings["data"]]
        return embeddings
