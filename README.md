# crowdannoDB

## For development

Clone the repo
```bash
git clone https://github.com/CrowdAnno/crowdannoDB.git
```

Start PostgreSQL DB
```bash
cd crowdannoDB/docker
docker-compose up
```

Run unit test with coverage report
```bash
coverage run -m unittest discover -s tests -p '*.py'
coverage report
```