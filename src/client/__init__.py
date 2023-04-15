from src.embedding.sentence_transformer import SentenceTransformerEmbedding
from src.mapper.postgres import Postgres
from src.mapper.sqlite import SQLite
from src.nearest_neighbors.hnswlib import HNSWNearestNeighbors
import logging

# Configure the root logger to output to the console
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class Client:
    def __init__(self, collection_name, url=None,
                 embedding_fn=SentenceTransformerEmbedding()):
        self.collection_id = None
        self.embedding_fn = embedding_fn

        # TODO: use default memory DB (e.g. SQLite)
        if url is None:
            self.db = SQLite()
        else:
            # TODO: per different url, construct different DB client
            self.db = Postgres(url)

        self.create_collection(collection_name)
        self.nearest_neighbors = None
        self.embeddings = self.db.get(collection_id=self.collection_id)

    def create_collection(self, name):
        self.collection_id, _ = self.db.create_collection(name)[0]
        return self.collection_id

    def get_collection(self, name):
        self.collection_id, _ = self.db.get_collection(name)
        return self.collection_id

    def add_data(self, texts, meta_data_list=None):
        embeddings = self.embedding_fn(texts)
        inserted_data = self.db.add(self.collection_id, embeddings, texts)
        # logger.info("inserted_data: %s", inserted_data)

        self.embeddings = self.db.get(self.collection_id)

        # TODO - update nearest neighbors search index more efficiently
        self.nearest_neighbors = HNSWNearestNeighbors(embeddings)

    def query(self, query_texts, top_k=3):
        if self.nearest_neighbors is None:
            self.embeddings = self.db.get(self.collection_id)
            embeddings = [row['embedding_data'] for row in self.embeddings]
            self.nearest_neighbors = HNSWNearestNeighbors(embeddings)
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        query_embeddings = self.embedding_fn(query_texts)
        indices = self.nearest_neighbors.search(query_embeddings, top_k)
        matched_embeddings = [{'text': r['text'], 'metadata': r['metadata']}
                              for i, r in enumerate(self.embeddings)
                              if i in indices]
        return matched_embeddings
