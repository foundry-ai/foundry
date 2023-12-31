# syntax = edrevo/dockerfile-plus
INCLUDE+ docker/Dockerfile.base

INCLUDE+ docker/Dockerfile.jax

FROM base as repo
COPY --from=jax-build /wheels/* /wheels

FROM base as stanza