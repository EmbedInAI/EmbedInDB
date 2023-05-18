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

import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from embedin.model.collection_model import CollectionModel, Base

engine = create_engine("sqlite:///:memory:", echo=True)
Session = sessionmaker(bind=engine)


class TestCollectionModel(unittest.TestCase):
    def setUp(self):
        Base.metadata.create_all(engine)
        self.session = Session()

    def tearDown(self):
        self.session.rollback()
        self.session.close()
        Base.metadata.drop_all(engine)

    def test_collection_model(self):
        # Create a new collection
        collection = CollectionModel(id="1", name="test")
        self.session.add(collection)
        self.session.commit()

        # Retrieve the collection from the database
        retrieved_collection = (
            self.session.query(CollectionModel).filter_by(id="1").one()
        )

        # Check that the retrieved collection matches the original collection
        self.assertEqual(collection.id, retrieved_collection.id)
        self.assertEqual(collection.name, retrieved_collection.name)

    def test_duplicate_name(self):
        # Create a new collection with a duplicate name
        collection1 = CollectionModel(id="1", name="test")
        collection2 = CollectionModel(id="2", name="test")

        # Add the first collection to the database
        self.session.add(collection1)
        self.session.commit()

        # Try to add the second collection to the database
        with self.assertRaises(Exception):
            self.session.add(collection2)
            self.session.commit()

        # Roll back the session to clear the transaction
        self.session.rollback()

        # Check that the second collection was not added to the database
        self.assertEqual(
            self.session.query(CollectionModel).filter_by(name="test").count(), 1
        )


if __name__ == "__main__":
    unittest.main()
