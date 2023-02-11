FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
  python-is-python3 python3-pip fish \
  && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY poetry.toml /code/poetry.toml
COPY poetry.lock /code/poetry.lock
COPY pyproject.toml /code/pyproject.toml

WORKDIR /code
RUN poetry install

COPY . /code
CMD ["poetry", "run", "python", "-m", "stanza.pool:worker"]