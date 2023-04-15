import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime

from src.mapper import DB

logger = logging.getLogger(__name__)


class SQLite(DB):
    def __init__(self, db_file=':memory:'):
        self.conn = sqlite3.connect(db_file)

    def get_collection(self, name):
        cursor = self.conn.cursor()
        query = "SELECT id, name FROM collection WHERE name = ?"
        cursor.execute(query, (name,))
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def _create_tables(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding (
                    id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding_data JSON NOT NULL,
                    metadata JSON NULL,
                    hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='index' AND name='unique_hash' AND tbl_name='embedding'
            """)
            constraint_exists = cursor.fetchone() is not None

            if not constraint_exists:
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS unique_hash ON embedding (hash)
                """)
        except Exception as e:
            print("Error: ", e)
        finally:
            cursor.close()

    def create_collection(self, name, get_if_exist=True):
        self._create_tables()
        collection = self.get_collection(name)
        if len(collection) > 0:
            if get_if_exist:
                return collection
            raise ValueError(f"Collection with name {name} already exists")

        collection_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        query = "INSERT INTO collection (id, name) VALUES (?, ?)"
        cursor.execute(query, (collection_id, name))
        self.conn.commit()
        cursor.close()
        return [(collection_id, name)]

    def add(self, collection_id, embeddings, texts, metadata_list=None):
        cur = self.conn.cursor()
        # Generate a list of tuples for executemany
        rows = []
        for i, embedding in enumerate(embeddings):
            # Generate a UUID for the embedding
            emb_id = str(uuid.uuid4())

            meta_data = metadata_list[i] if metadata_list else None
            data = texts[i] + collection_id + json.dumps(meta_data)

            hashed = hashlib.sha256(data.encode()).hexdigest()

            # Construct a row tuple with the columns of the embedding table
            row = (
                emb_id,
                collection_id,
                texts[i],
                json.dumps(embedding),
                json.dumps(meta_data),
                hashed,
                datetime.now()
            )
            rows.append(row)

        # Construct the SQL statement with placeholders for multiple rows
        placeholders = ",".join(["?"] * len(rows[0]))

        sql = f"""
        INSERT OR IGNORE INTO embedding (id, collection_id, text, embedding_data, metadata, hash, created_at)
        VALUES ({placeholders})
        """

        try:
            # Use executemany to insert the rows in bulk
            cur.executemany(sql, rows)

            self.conn.commit()
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            logger.error("Error: Duplicate hash values detected: %s",
                         str(e))

        # Get the successfully inserted data
        ids = [row[0] for row in rows]
        placeholders = ','.join(['?'] * len(ids))
        query = f"""
            SELECT id, collection_id, text, embedding_data, metadata, created_at
            FROM embedding
            WHERE id IN ({placeholders}) 
        """
        cur.execute(query, ids)
        inserted_rows = cur.fetchall()

        cur.close()
        return inserted_rows

    def get(self, collection_id=None, collection_name=None):
        with self.conn:
            if collection_name:
                # Retrieve the collection_id for the specified collection_name
                cur = self.conn.cursor()
                cur.execute("SELECT id FROM collection WHERE name = ?",
                            (collection_name,))
                collection_id = cur.fetchone()
                if not collection_id:
                    raise ValueError(
                        f"No collection found with name {collection_name}")
                collection_id = collection_id[0]
                cur.close()

            # Retrieve the embeddings for the specified collection_id
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, collection_id, text, embedding_data, metadata, created_at FROM embedding WHERE collection_id = ?",
                (collection_id,))
            rows = cur.fetchall()
            cur.close()

            # Construct a list of dictionaries for the embeddings
            embeddings = []
            for row in rows:
                embedding = {
                    "id": row[0],
                    "collection_id": row[1],
                    "text": row[2],
                    "embedding_data": json.loads(row[3]),
                    "metadata": row[4],
                    "created_at": row[5]
                }
                embeddings.append(embedding)

            return embeddings
