"""
OpenAI-compatible embedding provider for mcp-server-qdrant.

Supports any endpoint that implements the OpenAI /v1/embeddings API,
including: OpenAI, Azure OpenAI, local LM Studio, Ollama (openai mode),
9router, llama.cpp server, text-embeddings-inference, and more.

Configuration via environment variables:
  EMBEDDING_PROVIDER=openai
  EMBEDDING_MODEL=text-embedding-3-small       # or any model name
  OPENAI_API_KEY=sk-...                         # or EMBEDDING_API_KEY
  OPENAI_BASE_URL=https://api.openai.com/v1     # or any compatible URL
  EMBEDDING_VECTOR_SIZE=1536                    # required: must match model output
"""

from __future__ import annotations

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider that calls any OpenAI-compatible /v1/embeddings endpoint.

    Works with OpenAI, Azure OpenAI, local LM Studio, Ollama, llama.cpp,
    text-embeddings-inference, 9router, and any other OpenAI-compatible server.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        vector_size: int,
    ) -> None:
        """
        :param model_name:  Model identifier passed to the API (e.g. "text-embedding-3-small").
        :param base_url:    Base URL of the OpenAI-compatible endpoint, e.g.
                            "https://api.openai.com/v1" or "http://localhost:1234/v1".
        :param api_key:     Bearer token / API key for the endpoint.
        :param vector_size: Output dimensionality of the model.  Must be set
                            explicitly because we cannot introspect it without
                            making a live request.
        """
        self._model_name = model_name
        # Normalise: strip trailing slash so we can always append /embeddings
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._vector_size = vector_size

    # ── EmbeddingProvider interface ──────────────────────────────────────────

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents (passage-side)."""
        return await self._embed(documents)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self._embed([query])
        return results[0]

    def get_vector_name(self) -> str:
        """
        Vector name used inside the Qdrant collection.

        Returns the empty string so Qdrant uses the default unnamed vector,
        matching the behaviour of FastEmbedProvider and keeping collections
        compatible across provider switches.
        """
        return ""

    def get_vector_size(self) -> int:
        return self._vector_size

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {"model": self._model_name, "input": texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        # Respect the index field so batch ordering is guaranteed
        sorted_items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_items]
