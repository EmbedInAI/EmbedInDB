import uuid

import psycopg2

from src.mapper import DB
from datetime import datetime


class Postgres(DB):
    def __init__(self, config):
        self.conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            dbname=config.dbname
        )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_collection(self, name):
        with self.conn.cursor() as cursor:
            query = "SELECT id, name FROM collection WHERE name = %s"
            cursor.execute(query, (name,))
            rows = cursor.fetchall()
            return rows

    def _create_tables(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)

            cursor.execute("""
                            CREATE TABLE IF NOT EXISTS embedding (
                                id UUID PRIMARY KEY,
                                collection_id UUID NOT NULL,
                                text TEXT NOT NULL,
                                embedding FLOAT[] NOT NULL,
                                metadata JSONB NULL,
                                created_at TIMESTAMP NOT NULL DEFAULT NOW()
                            )
                        """)
            self.conn.commit()

    def create_collection(self, name, get_if_exist=True):
        self._create_tables()
        collection = self.get_collection(name)
        if len(collection) > 0:
            if get_if_exist:
                return collection
            raise ValueError(f"Collection with name {name} already exists")

        collection_id = str(uuid.uuid4())
        with self.conn.cursor() as cursor:
            query = "INSERT INTO collection (id, name) VALUES (%s, %s)"
            cursor.execute(query, (collection_id, name))
            self.conn.commit()
            return [(collection_id, name)]

    def add(self, collection_id, embeddings, texts, metadata_list=None):
        with self.conn.cursor() as cur:
            # Generate a list of tuples for executemany
            rows = []
            for i, emb in enumerate(embeddings):
                # Generate a UUID for the embedding
                emb_id = str(uuid.uuid4())

                # Construct a row tuple with the columns of the embedding table
                row = (
                    emb_id,
                    collection_id,
                    texts[i],
                    emb,
                    metadata_list[i] if metadata_list else None,
                    datetime.now()
                )
                rows.append(row)

            # Construct the SQL statement with placeholders for multiple rows
            placeholders = ",".join(["%s"] * len(rows[0]))
            sql = f"""
            INSERT INTO embedding (id, collection_id, text, embedding, metadata, created_at)
            VALUES ({placeholders})
            """

            # Print out the SQL statement and the rows being executed
            print(f"Executing SQL statement: {sql}")
            for row in rows:
                print(f"Inserting row: {row}")

            # Use executemany to insert the rows in bulk
            cur.executemany(sql, rows)

    def get(self, collection_name=None, collection_id=None):
        with self.conn.cursor() as cur:
            if collection_name:
                # Retrieve the collection_id for the specified collection_name
                cur.execute("SELECT id FROM collection WHERE name = %s",
                            (collection_name,))
                collection_id = cur.fetchone()
                if not collection_id:
                    raise ValueError(
                        f"No collection found with name {collection_name}")
                collection_id = collection_id[0]

            # Retrieve the embeddings for the specified collection_id
            cur.execute(
                "SELECT id, collection_id, text, embedding, metadata, created_at FROM embedding WHERE collection_id = %s",
                (collection_id,))
            rows = cur.fetchall()

            # Construct a list of dictionaries for the embeddings
            embeddings = []
            for row in rows:
                embedding = {
                    "id": row[0],
                    "collection_id": row[1],
                    "text": row[2],
                    "embedding": row[3],
                    "metadata": row[4],
                    "created_at": row[5]
                }
                embeddings.append(embedding)

            return embeddings
