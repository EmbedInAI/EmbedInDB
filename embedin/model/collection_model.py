from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CollectionModel(Base):
    __tablename__ = "collection"

    id = Column(String(36), primary_key=True)
    name = Column(String(64), unique=True)
