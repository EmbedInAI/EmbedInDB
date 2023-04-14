import os

from src.client import Client

if __name__ == '__main__':
    db_url = os.environ.get('DATABASE_URL', "postgres://postgres:postgres@localhost:5432/postgres")

    client = Client(collection_name="test_collection", url=db_url)
    client.add_data(["This is a test", "I like LLM2", "Hello world!"])
    result = client.query("These are tests", top_k=1)
    print("result: ", result)
