# -*- coding: utf-8 -*-
# embedin - A vector database that empowers AI with persistent memory,
# (C) 2023 EmbedInAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
