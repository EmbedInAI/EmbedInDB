import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from embedin.embedding.sentence_transformer import SentenceTransformerEmbedding
from embedin.index.hnswlib import HNSWNearestNeighbors
from embedin.service.collection_service import CollectionService
from embedin.service.embedding_service import EmbeddingService

# Configure the root logger to output to the console
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, collection_name, url=None,
                 embedding_fn=SentenceTransformerEmbedding(), debug=False):
        self.collection_id = None
        self.embedding_fn = embedding_fn

        if url is None:
            engine = create_engine('sqlite:///:memory:', echo=debug)
        else:
            engine = create_engine(url)

        session = sessionmaker(bind=engine)
        self.session = session()

        self.collection_service = CollectionService(self.session)
        self.embedding_service = EmbeddingService(self.session)

        collection_id = self.create_or_get_collection(collection_name)
        self.embeddings = self.embedding_service.get_by_collection_id(collection_id)
        embeddings = [row.embedding_data for row in self.embeddings]

        self.nearest_neighbors = None
        if embeddings:
            self.nearest_neighbors = HNSWNearestNeighbors(embeddings)

    def create_or_get_collection(self, name):
        collection_id = self.get_collection(name)
        if not collection_id:
            collection_id = self.create_collection(name)
        self.collection_id = collection_id
        return collection_id

    def create_collection(self, name):
        collection = self.collection_service.create(name)
        self.collection_id = collection.id
        return self.collection_id

    def get_collection(self, name):
        collection = self.collection_service.get_by_name(name)
        if collection:
            self.collection_id = collection.id
            return self.collection_id

    def add_data(self, texts, meta_data=None):
        embeddings = self.embedding_fn(texts)
        inserted_data = self.embedding_service.add_all(self.collection_id, embeddings, texts, meta_data)
        logger.info("inserted_data: %s", inserted_data)

        self.embeddings += inserted_data

        inserted_embeddings = [row.embedding_data for row in inserted_data]

        if self.nearest_neighbors is None:
            self.nearest_neighbors = HNSWNearestNeighbors(embeddings)
        else:
            self.nearest_neighbors.update_index(inserted_embeddings)

    def query(self, query_texts, top_k=3):
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        query_embeddings = self.embedding_fn(query_texts)
        indices = self.nearest_neighbors.search(query_embeddings, top_k)
        matched_embeddings = [{'text': r.text, 'meta_data': r.meta_data}
                              for i, r in enumerate(self.embeddings)
                              if i in indices]
        return matched_embeddings
