# -*- coding: utf-8 -*-
# embedin - A vector database that empowers AI with persistent memory,
# (C) 2023 EmbedInAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import logging
import uuid
from datetime import datetime

from embedin.repository.embedding_repository import EmbeddingRepository


# Configure the root logger to output to the console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


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
                List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.

        get_by_collection_id(collection_id):
            Fetches all embeddings from the database for a specified collection ID.

            Args:
                collection_id (str): The ID of the collection to retrieve embeddings for.

            Returns:
                List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
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
            List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
        """

        # Generate a list of Embedding objects
        rows = []
        hash_value_set = set()
        for i, embedding in enumerate(embeddings):
            # Generate a UUID for the embedding
            emb_id = str(uuid.uuid4())

            meta_data = metadata_list[i] if metadata_list else None
            data = texts[i] + collection_id + json.dumps(meta_data)

            hashed = hashlib.sha256(data.encode()).hexdigest()
            if hashed in hash_value_set:
                continue
            hash_value_set.add(hashed)

            # Construct an Embedding object
            row = dict(
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

        logger.info(f"{len(inserted_rows)} rows inserted")

        return inserted_rows

    def get_by_collection_id(self, collection_id):
        """
        Fetches all embeddings from the database for a specified collection ID.

        Args:
            collection_id (str): The ID of the collection to retrieve embeddings for.

        Returns:
            List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
        """

        # Get the Embedding objects for the specified collection_id
        rows = self.embedding_repo.get_by_collection_id(collection_id)

        return rows
