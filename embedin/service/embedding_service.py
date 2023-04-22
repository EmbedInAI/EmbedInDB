import hashlib
import json
import uuid
from datetime import datetime

from embedin.model.embedding_model import EmbeddingModel
from embedin.repository.embedding_repository import EmbeddingRepository


class EmbeddingService:
    def __init__(self, session):
        self.embedding_repo = EmbeddingRepository(session)

    def add_all(self, collection_id, embeddings, texts, metadata_list=None):
        # Generate a list of Embedding objects
        rows = []
        for i, embedding in enumerate(embeddings):
            # Generate a UUID for the embedding
            emb_id = str(uuid.uuid4())

            meta_data = metadata_list[i] if metadata_list else None
            data = texts[i] + collection_id + json.dumps(meta_data)

            hashed = hashlib.sha256(data.encode()).hexdigest()

            # Construct an Embedding object
            # TODO: should not call model class directly in service class
            row = EmbeddingModel(
                id=emb_id,
                collection_id=collection_id,
                text=texts[i],
                embedding_data=embedding,  # json.dumps(embedding),
                meta_data=meta_data,
                hash=hashed,
                created_at=datetime.now()
            )
            rows.append(row)

        # Add the Embedding objects to the session and commit the transaction
        inserted_rows = self.embedding_repo.add_all(rows)

        return inserted_rows

    def get_by_collection_id(self, collection_id):
        # Get the Embedding objects for the specified collection_id
        rows = self.embedding_repo.get_by_collection_id(collection_id)

        return rows
