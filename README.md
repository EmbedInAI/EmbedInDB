# Embedin - Empower AI with persistent memory
[![PyPI](https://img.shields.io/pypi/v/embedin?label=embedin)](https://pypi.org/project/embedin/)
[![Coverage Status](https://coveralls.io/repos/github/EmbedInAI/EmbedInDB/badge.svg)](https://coveralls.io/github/EmbedInAI/EmbedInDB)

Embedin is an open-source vector database and efficient library that seamlessly converts popular databases like MySQL, PostgreSQL, and MS SQL Server into vector databases with zero effort.

Embedin is an ideal solution for AI applications like natural language processing, image recognition, and recommendation systems, offering fast indexing and retrieval. Its simple API and query language ensure ease of use and seamless integration.

## Minimum Requirements
Python 3.7 or higher.

## Installation
```bash
pip install embedin
```

## Quick Start
### Store embeddings in memory
```python
from embedin import Embedin

client = Embedin(collection_name="test_collection", texts=["This is a test", "Hello world!"])
result = client.query("These are tests", top_k=1)  # Query the most similar text from the collection
print(result)
```

### Store embeddings in SQLite
```python
from embedin import Embedin

url = 'sqlite:///test.db'
client = Embedin(collection_name="test_collection", texts=["This is a test", "Hello world!"], url=url)
result = client.query("These are tests", top_k=1)
```

### Store embeddings in PostgreSQL

#### Start PostgreSQL Docker container
```bash
cd docker
docker-compose up embedin-postgres
```
__example__

```python
import os

from embedin import Embedin

url = os.getenv('EMBEDIN_POSGRES_URL', "postgresql+psycopg2://embedin:embedin@localhost/embedin_db")
client = Embedin(collection_name="test_collection", texts=["This is a test", "Hello world!"], url=url)
result = client.query("These are tests", top_k=1)
```

### Store embeddings in MySQL

#### Start MySQL Docker container
```bash
cd docker
docker-compose up embedin-mysql
```

__example__

```python
import os

from embedin import Embedin

url = os.getenv('EMBEDIN_MYSQL_URL', "mysql+pymysql://embedin:embedin@localhost/embedin_db")
client = Embedin(collection_name="test_collection", texts=["This is a test", "Hello world!"], url=url)
result = client.query("These are tests", top_k=1)
```

### Store embeddings in SQL Server

#### Start SQL Server Docker container
```bash
cd docker
docker-compose up embedin-mssql
```

__example__

```python
import os

from embedin import Embedin

url = os.getenv('EMBEDIN_MSSQL_URL', "mssql+pymssql://sa:StrongPassword123@localhost/tempdb")
client = Embedin(collection_name="test_collection", url=url)
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)
```

## More References
* [API Reference](./API.md)


## Contribute
Please refer [Contributors Guide](./CONTRIBUTING.md)