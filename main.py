import os

from src.client import Client

if __name__ == '__main__':
    sqlite_url = 'sqlite:///test.db'
    postgres_url = os.environ.get('EMBEDIN_POSGRES_URL', "postgresql+psycopg2://embedin:embedin@localhost/embedin_db")
    mysql_url = os.environ.get('EMBEDIN_MYSQL_URL', "mysql+pymysql://embedin:embedin@localhost/embedin_db")
    mssql_url = os.environ.get('EMBEDIN_MSSQL_URL', "mssql+pymssql://sa:StrongPassword123@localhost/tempdb")
    # oracle_url = os.environ.get('EMBEDIN_ORACLE_URL', "oracle+oracledb://system:password@localhost/ORCLCDB")

    # use sqlite in-memory
    client = Client(collection_name="test_collection")
    client.add_data(texts=["This is a test"],
                    meta_data=[{"source": "abc4"}])
    result = client.query("These are tests", top_k=1)

    print("result: ", result)
