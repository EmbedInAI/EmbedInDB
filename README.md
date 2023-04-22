# Embedin

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