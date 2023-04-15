import os

from src.client import Client

if __name__ == '__main__':
    # url = os.environ.get('DATABASE_URL', "postgres://postgres:postgres@localhost:5432/postgres")

    # use sqlite in-memory
    client = Client(collection_name="test_collection")
    client.add_data(["This is a test", "I like LLM2", "Hello world!"])
    result = client.query("These are tests", top_k=1)

    print("result: ", result)
