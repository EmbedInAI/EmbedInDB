from abc import ABC, abstractmethod


class DB(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_collection(self, name, get_if_exist=False):
        pass

    @abstractmethod
    def get_collection(self, name):
        pass

    @abstractmethod
    def add(self, collection_id, embeddings, texts, metadata_list=None):
        pass

    @abstractmethod
    def get(self, collection_name=None, collection_id=None):
        pass
