FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

EXPOSE 8000

ENV QDRANT_URL="" \
    QDRANT_API_KEY="" \
    COLLECTION_NAME="" \
    EMBEDDING_PROVIDER="fastembed" \
    EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    OPENAI_BASE_URL="https://api.openai.com/v1" \
    OPENAI_API_KEY="" \
    EMBEDDING_VECTOR_SIZE="1536"

CMD ["uv", "run", "mcp-server-qdrant", "--transport", "streamable-http"]
