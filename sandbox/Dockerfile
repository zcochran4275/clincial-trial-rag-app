# Minimal image for running agent-generated extraction code + tests.
# No network at runtime (enforced by `docker run --network none` in agent/sandbox.py).
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.* pydantic==2.*
WORKDIR /work
