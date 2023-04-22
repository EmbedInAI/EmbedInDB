import uuid

from embedin.model.collection_model import CollectionModel


class CollectionRepository:
    def __init__(self, session):
        self.session = session
        self.create_table()

    def create_table(self):
        CollectionModel.metadata.create_all(self.session.bind)

    def get_by_name(self, name):
        collection = self.session.query(CollectionModel).filter_by(name=name).first()
        return collection

    def create(self, name, get_if_exist=True):
        collection = self.get_by_name(name)
        if collection:
            if get_if_exist:
                return collection
            raise ValueError(f"Collection with name {name} already exists")

        collection_id = str(uuid.uuid4())
        collection = CollectionModel(id=collection_id, name=name)
        self.session.add(collection)
        self.session.commit()

        return collection
