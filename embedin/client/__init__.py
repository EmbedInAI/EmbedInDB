import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from embedin.embedding import Embedding
from embedin.index import Index
from embedin.service.collection_service import CollectionService
from embedin.service.embedding_service import EmbeddingService

# Configure the root logger to output to the console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class Client:
    """
    Client class for managing embeddings and performing nearest neighbor search.

    Args:
        collection_name (str): Name of the collection.
        url (str, optional): Database URL. Defaults to None.
        embedding_fn (str or EmbeddingFunction, optional): The embedding function to use. The default is the SentenceTransformer model. Supported options are 'sentence_transformer' and 'openai'. If 'openai' is selected, the 'OPENAI_API_KEY' environment variable must be set to authenticate with the OpenAI API.
        index_hint (str, optional): Similarity search index to use. Supports: 'flat' and 'hnsw'
        debug (bool, optional): Enable debug mode. Defaults to False.

    Attributes:
        collection_id (int): ID of the collection.
        embedding_fn (str or EmbeddingFunction): Embedding function to use. Support 'sentence_transformer' and 'openai'.
        session (Session): SQLAlchemy session.
        collection_service (CollectionService): CollectionService instance.
        embedding_service (EmbeddingService): EmbeddingService instance.
        embedding_rows (List[EmbeddingModel]): List of Embedding instances for the current collection.

    Methods:
        create_or_get_collection(name): Get the ID of an existing collection or create a new one.
        create_collection(name): Create a new collection with the given name.
        get_collection(name): Get the ID of an existing collection with the given name.
        add_data(texts, meta_data=None): Add new data to the collection.
        query(query_texts, top_k=3): Find nearest neighbors for the given query text(s).
    """

    def __init__(
        self,
        collection_name,
        url=None,
        embedding_fn="sentence_transformer",
        index_hint=None,
        debug=False,
        **kwargs
    ):
        self.collection_id = None

        if callable(embedding_fn):
            self.embedding_fn = embedding_fn
        else:
            self.embedding_fn = Embedding.create_embedding(embedding_fn)

        if url is None:
            engine = create_engine("sqlite:///:memory:", echo=debug)
        else:
            engine = create_engine(url)

        session = sessionmaker(bind=engine)
        self.session = session()

        self.collection_service = CollectionService(self.session)
        self.embedding_service = EmbeddingService(self.session)

        embeddings = kwargs.get("embeddings")
        embeddings_meta = kwargs.get("meta")
        texts = kwargs.get("texts")

        self.collection_id = self.create_or_get_collection(collection_name)
        self.embedding_rows = self.embedding_service.get_by_collection_id(
            self.collection_id
        )

        # Insert user-provided texts and embeddings into the database
        inserted_rows = []
        if embeddings is not None and texts is not None:
            # TODO validate embeddings' shape and dtype; validate text as well
            inserted_rows = self.embedding_service.add_all(
                self.collection_id, embeddings, texts, embeddings_meta
            )
        elif texts is not None:
            embeddings = self.embedding_fn(texts)
            inserted_rows = self.embedding_service.add_all(
                self.collection_id, embeddings, texts, embeddings_meta
            )

        self.embedding_rows.extend(inserted_rows)
        total_embeddings = [row.embedding_data for row in self.embedding_rows]

        self.index_hint = index_hint
        self.search_index = None
        if total_embeddings:
            self.search_index = Index(total_embeddings, self.index_hint)

    def create_or_get_collection(self, name):
        """
        Get the ID of an existing collection or create a new one with the given name.

        Args:
            name (str): Name of the collection.

        Returns:
            str: ID of the collection.
        """
        collection_id = self.get_collection(name)
        if not collection_id:
            collection_id = self.create_collection(name)
        self.collection_id = collection_id
        return collection_id

    def create_collection(self, name):
        """
        Create a new collection with the given name.

        Args:
            name (str): Name of the collection.

        Returns:
            str: ID of the new collection.
        """
        collection = self.collection_service.create(name)
        self.collection_id = collection.id
        return self.collection_id

    def get_collection(self, name):
        """
        Retrieve the ID of an existing collection with the given name.

        Args:
            name (str): Name of the collection.

        Returns:
            str: ID of the collection, or None if no collection with the given name exists.
        """
        collection = self.collection_service.get_by_name(name)
        if collection:
            self.collection_id = collection.id
            return self.collection_id

    def add_data(self, texts, meta_data=None):
        """
        Add new data to the collection.

        Args:
            texts (list of str): List of text data to add to the collection.
            meta_data (list or None): List of associated metadata for the added text data (optional).

        Returns:
            None.
        """
        embeddings = self.embedding_fn(texts)
        logger.info("Adding embedding to the database")
        inserted_data = self.embedding_service.add_all(
            self.collection_id, embeddings, texts, meta_data
        )

        self.embedding_rows.extend(inserted_data)

        inserted_embeddings = [row.embedding_data for row in inserted_data]

        if self.search_index is None:
            self.search_index = Index(embeddings, self.index_hint)
        else:
            self.search_index.update_index(inserted_embeddings)

    def query(self, query_texts, top_k=3):
        """
        Search for the most similar text data in the collection to the given query text.

        Args:
            query_texts (list or str): List of query text strings or a single query text string.
            top_k (int): The number of top matches to return (default is 3).

        Returns:
            list of dict: A list of dictionaries containing the most similar text data in the collection to the given query text.
                          Each dictionary has the following format:
                          {'text': str, 'meta_data': object or None}
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        query_embeddings = self.embedding_fn(query_texts)
        indices = self.search_index.search(query_embeddings, top_k)
        matched_embeddings = [
            {"text": r.text, "meta_data": r.meta_data}
            for i, r in enumerate(self.embedding_rows)
            if i in indices
        ]
        return matched_embeddings
