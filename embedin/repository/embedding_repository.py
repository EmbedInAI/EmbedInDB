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
import logging

from sqlalchemy.exc import IntegrityError
from tqdm import tqdm

from embedin.model.embedding_model import EmbeddingModel

# Configure the root logger to output to the console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """
    This class provides methods to interact with embeddings stored in the database.

    Attributes:
        session (sqlalchemy.orm.Session): The database session to use for querying and modifying data.

    Methods:
        create_table(): Creates the EmbeddingModel table in the database.
        add_all(rows): Adds multiple embeddings to the database, returning the successfully inserted rows.
        get_by_ids(ids): Returns the embeddings with the given ids.
        get_by_collection_id(collection_id): Returns all embeddings in the collection with the given id.
    """

    def __init__(self, session):
        """
        Constructs an EmbeddingRepository object.

        Args:
            session (sqlalchemy.orm.Session): The database session to use for querying and modifying data.
        """

        self.session = session
        self.create_table()

    def create_table(self):
        """
        Creates the EmbeddingModel table in the database.
        """

        EmbeddingModel.metadata.create_all(self.session.bind)

    def _add_rows_one_by_one(self, rows):
        """
        Adds multiple embeddings to the database using loop, returning the successfully inserted rows.

        Args:
            rows (List[EmbeddingModel]): A list of EmbeddingModel objects to add to the database.

        Returns:
            List[EmbeddingModel]: A list of successfully inserted EmbeddingModel.
        """
        inserted_rows = []
        for row in rows:
            try:
                self.session.merge(row)
                self.session.commit()
                inserted_rows.append(row)
            except (IntegrityError, Exception):
                self.session.rollback()
        return inserted_rows

    def add_all(self, rows):
        """
        The method adds multiple embeddings to the database in batches, and in case of exceptions,
        it switches to adding them one by one. It then returns the list of successfully inserted rows.

        Args:
            rows (List[dct]): A list of dict objects to add to the database.

        Returns:
            List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
        """
        rows = [EmbeddingModel(**row) for row in rows]

        inserted_rows = []

        batch_size = 200
        extra_batch = len(rows) % batch_size > 0
        num_batches = len(rows) // batch_size + extra_batch

        with tqdm(total=len(rows), desc="Inserting rows to database") as pbar:
            for batch_index in range(num_batches):
                batch_start = batch_index * batch_size
                batch_end = batch_start + batch_size
                batch = rows[batch_start:batch_end]

                try:
                    self.session.bulk_save_objects(batch)  # TODO using SQL
                    self.session.commit()
                    inserted_rows.extend(batch)

                except (IntegrityError, Exception) as e:
                    self.session.rollback()
                    self.session.expunge_all()
                    inserted_rows.extend(self._add_rows_one_by_one(batch))

                pbar.update(len(batch))

        inserted_rows = [r.to_dict() for r in inserted_rows]

        return inserted_rows

    # This is only needed when bulk add in add_all is implemented
    def get_by_ids(self, ids):
        """
        Returns the embeddings with the given ids.

        Args:
            ids (List[str]): A list of embedding ids to retrieve.

        Returns:
            List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
        """

        # Get the successfully inserted data
        rows = (
            self.session.query(EmbeddingModel).filter(EmbeddingModel.id.in_(ids)).all()
        )

        rows = [r.to_dict() for r in rows]

        return rows

    def get_by_collection_id(self, collection_id):
        """
        Returns all embeddings in the collection with the given id.

        Args:
            collection_id (str): The id of the collection to retrieve embeddings from.

        Returns:
            List[dict]: A list of dictionaries representing EmbeddingModel objects from the collection with the given id.
        """

        rows = (
            self.session.query(EmbeddingModel)
            .filter(EmbeddingModel.collection_id == collection_id)
            .all()
        )

        rows = [r.to_dict() for r in rows]

        return rows
