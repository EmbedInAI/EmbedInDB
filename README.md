# Embedin
[![Coverage Status](https://coveralls.io/repos/github/EmbedInAI/EmbedInDB/badge.svg)](https://coveralls.io/github/EmbedInAI/EmbedInDB)

## Installation
```bash
pip install embedin
```

## Quick Start
### Using memery DB
```python
from embedin.client import Client

client = Client(collection_name="test_collection")
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)

print("result: ", result)
```

### Using sqlite with local storage
```python
from embedin.client import Client

url = 'sqlite:///test.db'
client = Client(collection_name="test_collection", url=url)
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)
```

### Using PostgreSQL
```python
import os

from embedin.client import Client

url = os.environ.get('EMBEDIN_POSGRES_URL', "postgresql+psycopg2://embedin:embedin@localhost/embedin_db")
client = Client(collection_name="test_collection", url=url)
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)
```

### Using MySQL
```python
import os

from embedin.client import Client

url = os.environ.get('EMBEDIN_MYSQL_URL', "mysql+pymysql://embedin:embedin@localhost/embedin_db")
client = Client(collection_name="test_collection", url=url)
client.add_data(texts=["This is a test"], meta_data=[{"source": "abc4"}])
result = client.query("These are tests", top_k=1)
```

### Using MS-SQL
```python
import os

from embedin.client import Client

url = os.environ.get('EMBEDIN_MSSQL_URL', "mssql+pymssql://sa:StrongPassword123@localhost/tempdb")
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

Start Oracle DB (Only necessary for using Oracle as the storage) - not supported yet

docker login container-registry.oracle.com
```bash
cd docker
docker-compose up embedin-oracle
```yaml
  embedin-oracle:
    container_name: embedin-oracle
    image: container-registry.oracle.com/database/enterprise:21.3.0.0
    environment:
      - ORACLE_SID=ORCLCDB
      - ORACLE_PWD=password
      - ORACLE_PDB=ORCLPDB1
      - ORACLE_CHARACTERSET=AL32UTF8
    volumes:
      - ./oracle/data:/opt/oracle/oradata
    ports:
      - "1521:1521"
```

Run unit test with coverage report
```bash
coverage run -m unittest discover -s tests -p '*.py'
coverage report
```

To publish to PyPI
```bash
pip install twine
```

```bash
python setup.py sdist && python setup.py bdist_wheel && twine upload dist/*
```