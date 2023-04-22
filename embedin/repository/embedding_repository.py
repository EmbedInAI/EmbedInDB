from sqlalchemy.exc import IntegrityError

from embedin.model.embedding_model import EmbeddingModel


class EmbeddingRepository:
    def __init__(self, session):
        self.session = session
        self.create_table()

    def create_table(self):
        EmbeddingModel.metadata.create_all(self.session.bind)

    def add_all(self, rows):
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
        # Get the successfully inserted data
        rows = self.session.query(EmbeddingModel).filter(
            EmbeddingModel.id.in_(ids)).all()

        return rows

    def get_by_collection_id(self, collection_id):
        rows = self.session.query(EmbeddingModel).filter(
            EmbeddingModel.collection_id == collection_id).all()

        return rows
