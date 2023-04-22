import hashlib
import json
import uuid
from datetime import datetime

from embedin.model.embedding_model import EmbeddingModel
from embedin.repository.embedding_repository import EmbeddingRepository


class EmbeddingService:
    """
    A service class for handling operations related to embeddings.

    Attributes:
        embedding_repo (EmbeddingRepository): The repository instance for handling database interactions.

    Methods:
        add_all(collection_id, embeddings, texts, metadata_list=None):
            Adds multiple embeddings to the database.

            Args:
                collection_id (str): The ID of the collection that the embeddings belong to.
                embeddings (list): A list of embedding vectors, represented as numpy arrays.
                texts (list): A list of text strings that correspond to the embeddings.
                metadata_list (list, optional): A list of metadata objects, one for each embedding.

            Returns:
                list: A list of EmbeddingModel objects representing the newly created embeddings.

        get_by_collection_id(collection_id):
            Fetches all embeddings from the database for a specified collection ID.

            Args:
                collection_id (str): The ID of the collection to retrieve embeddings for.

            Returns:
                list: A list of EmbeddingModel objects representing the embeddings for the specified collection ID.
    """

    def __init__(self, session):
        """
        Initializes a new instance of the EmbeddingService class.

        Args:
            session (Session): A database session object for making database queries.
        """

        self.embedding_repo = EmbeddingRepository(session)

    def add_all(self, collection_id, embeddings, texts, metadata_list=None):
        """
        Adds multiple embeddings to the database.

        Args:
            collection_id (str): The ID of the collection that the embeddings belong to.
            embeddings (list): A list of embedding vectors, represented as numpy arrays.
            texts (list): A list of text strings that correspond to the embeddings.
            metadata_list (list, optional): A list of metadata objects, one for each embedding.

        Returns:
            list: A list of EmbeddingModel objects representing the newly created embeddings.
        """

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
                created_at=datetime.now(),
            )
            rows.append(row)

        # Add the Embedding objects to the session and commit the transaction
        inserted_rows = self.embedding_repo.add_all(rows)

        return inserted_rows

    def get_by_collection_id(self, collection_id):
        """
        Fetches all embeddings from the database for a specified collection ID.

        Args:
            collection_id (str): The ID of the collection to retrieve embeddings for.

        Returns:
            list: A list of EmbeddingModel objects representing the embeddings for the specified collection ID.
        """

        # Get the Embedding objects for the specified collection_id
        rows = self.embedding_repo.get_by_collection_id(collection_id)

        return rows
