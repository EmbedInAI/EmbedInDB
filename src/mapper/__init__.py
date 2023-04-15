from abc import ABC, abstractmethod


class DB(ABC):
    @abstractmethod
    def __init__(self):
        self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
