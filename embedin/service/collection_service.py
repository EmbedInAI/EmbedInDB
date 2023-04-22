from embedin.repository.collection_repository import CollectionRepository


class CollectionService:
    def __init__(self, session):
        self.collection_repo = CollectionRepository(session)

    def get_by_name(self, name):
        row = self.collection_repo.get_by_name(name)
        return row

    def create(self, name, get_if_exist=True):
        collection = self.collection_repo.create(name, get_if_exist)
        return collection
