FROM debian:11-slim AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-pip && \
    pip install pipenv

FROM build AS build-pipenv

ENV PIPENV_VENV_IN_PROJECT=1
ADD Pipfile.lock Pipfile /build/
WORKDIR /build
RUN pipenv sync

#Runtime
FROM gcr.io/distroless/python3-debian11 AS runtime

COPY --from=build-pipenv /build/.venv/ /app/.venv/
# Copy source files
COPY main.py /app
COPY train.py /app
COPY feature_store.yaml /app
WORKDIR /app
# Start application
ENTRYPOINT [ "/app/.venv/bin/python", "main.py"]