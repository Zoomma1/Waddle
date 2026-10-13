FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Deps layer (caché tant que pyproject/lock/src inchangés)
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY scripts ./scripts
COPY config ./config
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
