# Contributors Guide

``embedin`` welcomes your contribution. Below steps can be followed to create a pull request.

* Fork this embedin repository into your account.
* Clone the repo from your account `$ git clone https://github.com/<YOUR-ACCOUNT>/embedin.git`
* Create a feature branch in your fork (`$ git checkout -b my-new-feature`).
* Install dependencies (`$ pip install -r requirements.txt && pip install -r requirements-dev.txt`)
* Hack, hack, hack...
* Run unit test with coverage report `$ coverage run -m unittest discover -s tests -p '*.py' && coverage report`
* Run code style fix `$ black .`
* Commit your changes (`$ git commit -am 'Add some feature'`).
* Push the feature branch into your fork (`$ git push origin -u my-new-feature`).
* Create new pull request to `main` branch.
* Verify both unit test and code coverage check have passed in the pull request

## Start different databases in Docker for development

### Start PostgreSQL
```bash
cd docker
docker-compose up embedin-postgres
```

### Start MySQL
```bash
cd docker
docker-compose up embedin-mysql
```

### Start SQL Server
```bash
cd docker
docker-compose up embedin-mssql
```
