# Embedin - Empower AI with persistent memory
[![PyPI](https://img.shields.io/pypi/v/embedin?label=embedin)](https://pypi.org/project/embedin/)
[![Coverage Status](https://coveralls.io/repos/github/EmbedInAI/EmbedInDB/badge.svg)](https://coveralls.io/github/EmbedInAI/EmbedInDB)

## What is Embedin
Embedin is a highly efficient software library designed to seamlessly convert widely-used databases, such as MySQL, PostgreSQL, and MS SQL Server, into a vector database with minimal effort. Its lightweight design enables quick and easy integration into your existing software stack, allowing you to leverage the benefits of a vector database without the need for extensive modifications to your current system. 

With fast indexing and retrieval, it's ideal for high-performance applications such as natural language processing, image recognition, and recommendation systems. Embedin's simple API and query language make it easy to use and integrate. 

## Installation
```bash
pip install embedin
```

## Quick Start
### Using memory
```python
from embedin.client import Client

client = Client(collection_name="test_collection", texts=["This is a test", "Hello world!"])
result = client.query("These are tests", top_k=1)
print(result)
```

### Using Sqlite
```python
from embedin.client import Client

url = 'sqlite:///test.db'
client = Client(collection_name="test_collection", texts=["This is a test", "Hello world!"])
result = client.query("These are tests", top_k=1)
```

### Using PostgreSQL
```python
import os

from embedin.client import Client

url = os.getenv('EMBEDIN_POSGRES_URL', "postgresql+psycopg2://embedin:embedin@localhost/embedin_db")
client = Client(collection_name="test_collection", texts=["This is a test", "Hello world!"])
result = client.query("These are tests", top_k=1)
```

### Using MySQL
```python
import os

from embedin.client import Client

url = os.getenv('EMBEDIN_MYSQL_URL', "mysql+pymysql://embedin:embedin@localhost/embedin_db")
client = Client(collection_name="test_collection", texts=["This is a test", "Hello world!"])
result = client.query("These are tests", top_k=1)
```

### Using MS-SQL
```python
import os

from embedin.client import Client

url = os.getenv('EMBEDIN_MSSQL_URL', "mssql+pymssql://sa:StrongPassword123@localhost/tempdb")
client = Client(collection_name="test_collection", url=url)
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)
```

## For development

Clone the repo
```bash
git clone https://github.com/EmbedInAI/embedin.git
```

Start PostgreSQL DB (Only necessary for using PostgreSQL as the storage)
```bash
cd docker
docker-compose up embedin-postgres
```

Start MySQL DB (Only necessary for using MySQL as the storage)
```bash
cd docker
docker-compose up embedin-mysql
```

Start MS-SQL DB (Only necessary for using MS-SQL as the storage)
```bash
cd docker
docker-compose up embedin-mssql
```

Run unit test with coverage report
```bash
coverage run -m unittest discover -s tests -p '*.py'
coverage report
```