from sqlalchemy.exc import IntegrityError

from embedin.model.embedding_model import EmbeddingModel


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

    def add_all(self, rows):
        """
        Adds multiple embeddings to the database, returning the successfully inserted rows.

        Args:
            rows (List[EmbeddingModel]): A list of EmbeddingModel objects to add to the database.

        Returns:
            List[EmbeddingModel]: A list of successfully inserted EmbeddingModel objects.
        """

        # Add the Embedding objects to the session and commit the transaction
        inserted_rows = []
        for row in rows:
            try:
                self.session.merge(row)
                self.session.commit()
                inserted_rows.append(row)
            except IntegrityError:
                self.session.rollback()
        return inserted_rows

        # TODO: bulk add will rollback all rows in case of exception
        # try:
        #     self.session.add_all(rows)
        #     self.session.commit()
        #     self.session.flush()  # immediately flush to the database
        # except IntegrityError as e:
        #     # a session rollback will rollback only the failed rows
        #     self.session.rollback()
        #     # Remove any uncommitted objects from the session
        #     self.session.expunge_all()

    # This is only needed when bulk add in add_all is implemented
    def get_by_ids(self, ids):
        """
        Returns the embeddings with the given ids.

        Args:
            ids (List[str]): A list of embedding ids to retrieve.

        Returns:
            List[EmbeddingModel]: A list of EmbeddingModel objects with the given ids.
        """

        # Get the successfully inserted data
        rows = (
            self.session.query(EmbeddingModel).filter(EmbeddingModel.id.in_(ids)).all()
        )

        return rows

    def get_by_collection_id(self, collection_id):
        """
        Returns all embeddings in the collection with the given id.

        Args:
            collection_id (str): The id of the collection to retrieve embeddings from.

        Returns:
            List[EmbeddingModel]: A list of EmbeddingModel objects in the collection with the given id.
        """

        rows = (
            self.session.query(EmbeddingModel)
            .filter(EmbeddingModel.collection_id == collection_id)
            .all()
        )

        return rows
