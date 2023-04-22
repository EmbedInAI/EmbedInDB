import uuid

from embedin.model.collection_model import CollectionModel


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
        collection: CollectionModel or None
            The collection with the given name, or None if it does not exist.
        """

        collection = self.session.query(CollectionModel).filter_by(name=name).first()
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
        collection: CollectionModel
            The newly created collection.
        """

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
