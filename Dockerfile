FROM python:3.12-slim AS build

WORKDIR /app

COPY . .

RUN pip install uv --no-cache-dir && \
    uv sync --no-dev --frozen

FROM python:3.12-slim

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "src/agent.py", "start"]
