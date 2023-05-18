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

from embedin.repository.collection_repository import CollectionRepository


class CollectionService:
    """
    A service class for handling operations related to collections.

    Attributes:
        collection_repo (CollectionRepository): The repository instance for handling database interactions.

    Methods:
        get_by_name(name):
            Fetches a collection from the database by its name.

            Args:
                name (str): The name of the collection to retrieve.

            Returns:
                CollectionModel: A CollectionModel representing the row of the collection fetched from the database.

        create(name, get_if_exist=True):
            Creates a new collection in the database.

            Args:
                name (str): The name of the new collection.
                get_if_exist (bool): If True, return the existing collection if it already exists. If False, create a new collection with the given name.

            Returns:
                CollectionModel: A CollectionModel representing the newly created collection.
    """

    def __init__(self, session):
        """
        Initializes a new instance of the CollectionService class.

        Args:
            session (Session): A database session object for making database queries.
        """

        self.collection_repo = CollectionRepository(session)

    def get_by_name(self, name):
        """
        Fetches a collection from the database by its name.

        Args:
            name (str): The name of the collection to retrieve.

        Returns:
            CollectionModel: A CollectionModel representing the row of the collection fetched from the database.
        """

        row = self.collection_repo.get_by_name(name)
        return row

    def create(self, name, get_if_exist=True):
        """
        Creates a new collection in the database.

        Args:
            name (str): The name of the new collection.
            get_if_exist (bool): If True, return the existing collection if it already exists. If False, create a new collection with the given name.

        Returns:
            CollectionModel: A CollectionModel representing the newly created collection.
        """

        collection = self.collection_repo.create(name, get_if_exist)
        return collection
