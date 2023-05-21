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
import uuid

from sqlalchemy.exc import IntegrityError

from embedin.model.collection_model import CollectionModel

# Configure the root logger to output to the console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class CollectionRepository:
    """
    A repository for managing collections in a database.

    Parameters:
    -----------
    session: sqlalchemy.orm.Session
        The database session to use.

    Methods:
    --------
    create_table():
        Creates the table in the database for collections.

    get_by_name(name: str) -> CollectionModel or None:
        Retrieves a collection from the database by name.

    create(name: str, get_if_exist: bool = True) -> CollectionModel:
        Creates a new collection in the database with the given name.
        If the collection already exists and get_if_exist is True, it returns the existing collection.
        If get_if_exist is False, it raises a ValueError.
    """

    def __init__(self, session):
        """
        Constructs a new CollectionRepository with the given database session.
        """

        self.session = session
        self.create_table()

    def create_table(self):
        """
        Creates the table in the database for collections.
        """

        CollectionModel.metadata.create_all(self.session.bind)

    def get_by_name(self, name):
        """
        Retrieves a collection from the database by name.

        Parameters:
        -----------
        name: str
            The name of the collection to retrieve.

        Returns:
        --------
        collection: dict
            The collection with the given name,
        """

        collection = self.session.query(CollectionModel).filter_by(name=name).first()

        collection = collection.to_dict() if collection else {}
        return collection

    def create(self, name, get_if_exist=True):
        """
        Creates a new collection in the database with the given name.

        Parameters:
        -----------
        name: str
            The name of the collection to create.
        get_if_exist: bool, optional (default=True)
            If the collection already exists and get_if_exist is True, it returns the existing collection.
            If get_if_exist is False, it raises a ValueError.

        Returns:
        --------
        collection: dict
            The newly created collection.
        """

        collection = self.get_by_name(name)
        if collection:
            if get_if_exist:
                return collection
            raise ValueError(f"Collection with name {name} already exists")

        collection_id = str(uuid.uuid4())
        collection_model = CollectionModel(id=collection_id, name=name)

        try:
            self.session.add(collection_model)
            self.session.commit()
            return collection_model.to_dict()
        except (IntegrityError, Exception) as e:
            self.session.rollback()
            logger.error(f"creating collection encounter an error: {str(e)}")
