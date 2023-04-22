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
                dict: A dictionary representing the row of the collection fetched from the database.

        create(name, get_if_exist=True):
            Creates a new collection in the database.

            Args:
                name (str): The name of the new collection.
                get_if_exist (bool): If True, return the existing collection if it already exists. If False, create a new collection with the given name.

            Returns:
                dict: A dictionary representing the newly created collection.
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
            dict: A dictionary representing the row of the collection fetched from the database.
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
            dict: A dictionary representing the newly created collection.
        """

        collection = self.collection_repo.create(name, get_if_exist)
        return collection
