FROM base as numpy-build

RUN pip install Cython
RUN pip wheel --no-binary numpy numpy=={{config.custom.numpy.version}}