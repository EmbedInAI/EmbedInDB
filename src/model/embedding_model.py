from sqlalchemy import Column, String, Text, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EmbeddingModel(Base):
    __tablename__ = "embedding"

    id = Column(String(36), primary_key=True)
    collection_id = Column(String(36))
    text = Column(Text)

    # TODO: Only Postgres supports ARRAY(Float). Using JSON for now
    # https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.ARRAY
    embedding_data = Column(JSON)

    meta_data = Column(JSON)
    hash = Column(String(64), unique=True)
    created_at = Column(DateTime)
