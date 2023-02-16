FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

## preesed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select America" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/America select New_York" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg; \
    rm -f /etc/timezone /etc/localtime; \
    apt-get update; \
    apt-get upgrade; \
    apt-get install -y software-properties-common; \
    apt-add-repository ppa:fish-shell/release-3; \
    apt-get update; \
    apt-get install -y tzdata && \
    apt-get install -y \
        build-essential wget curl \
        bash python3 python3-pip python-is-python3 fish git neovim \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY poetry.toml /code/poetry.toml
COPY poetry.lock /code/poetry.lock
COPY pyproject.toml /code/pyproject.toml

WORKDIR /code
RUN poetry config installer.max-workers 10
RUN poetry install --no-interaction --no-ansi -vvv

# Setup entrypoint
RUN echo "#!/usr/bin/env bash\nexec poetry run \"\$@\"" > /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV TERM=xterm
ENV SHELL=fish
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]

COPY . /code
