"""Embedding providers — generate text embeddings via LLM provider APIs.

Supports OpenAI, Google, and Ollama embedding endpoints.
No new dependencies — uses the SDK clients already installed.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (list of floats).
        """

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Query text.
            
        Returns:
            Embedding vector.
        """
        results = self.embed([text])
        return results[0] if results else []

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""


class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings via OpenAI API (text-embedding-3-small).
    
    Also works with any OpenAI-compatible endpoint (Groq, Mistral, etc.)
    that supports the /v1/embeddings endpoint.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._model = model
        self._dimension = 1536  # text-embedding-3-small default

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI embeddings API."""
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning("OpenAI embedding failed: %s", exc)
            return []

    @property
    def dimension(self) -> int:
        return self._dimension


class GoogleEmbeddings(EmbeddingProvider):
    """Embeddings via Google Gemini API."""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Google embedding API."""
        if not texts:
            return []
        results: list[list[float]] = []
        try:
            for text in texts:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=text,
                )
                if hasattr(response, 'embedding') and response.embedding:
                    # response.embedding could be an object with 'values'
                    emb = response.embedding
                    if hasattr(emb, 'values'):
                        results.append(list(emb.values))
                    elif isinstance(emb, list):
                        results.append(emb)
                    else:
                        results.append([])
                else:
                    results.append([])
            return results
        except Exception as exc:
            logger.warning("Google embedding failed: %s", exc)
            return []

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings via Ollama local API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = 768  # nomic-embed-text default

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama embeddings API.

        Tries the new ``/api/embed`` endpoint first (Ollama 0.4+),
        falls back to the legacy ``/api/embeddings`` endpoint.
        Failures are silent (debug-level) since embeddings are optional.
        """
        if not texts:
            return []
        import httpx
        results: list[list[float]] = []
        try:
            # New endpoint (Ollama 0.4+): /api/embed with "input" field
            resp = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                if not self._dimension and embeddings[0]:
                    self._dimension = len(embeddings[0])
                return embeddings
        except httpx.HTTPStatusError:
            # Fall back to legacy /api/embeddings (one at a time)
            pass
        except Exception as exc:
            logger.debug("Ollama embedding (new API) unavailable: %s", exc)
            return []

        # Legacy endpoint fallback
        try:
            for text in texts:
                resp = httpx.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding", [])
                results.append(embedding)
                if not self._dimension and embedding:
                    self._dimension = len(embedding)
            return results
        except Exception as exc:
            logger.debug("Ollama embedding unavailable: %s", exc)
            return []

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(
    api_keys: dict[str, str],
) -> Optional[EmbeddingProvider]:
    """Factory: pick the best available embedding provider.
    
    Priority: OpenAI > Google > Ollama (always available).
    
    Args:
        api_keys: Dict of provider name -> API key.
        
    Returns:
        An EmbeddingProvider instance, or None if nothing works.
    """
    # OpenAI (best quality embeddings)
    if api_keys.get("openai"):
        try:
            return OpenAIEmbeddings(api_key=api_keys["openai"])
        except Exception as exc:
            logger.debug("OpenAI embeddings unavailable: %s", exc)

    # Google
    if api_keys.get("google"):
        try:
            return GoogleEmbeddings(api_key=api_keys["google"])
        except Exception as exc:
            logger.debug("Google embeddings unavailable: %s", exc)

    # Ollama (local — only if an embedding model is actually pulled)
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            # Check for known embedding models (ordered by quality — best first)
            embedding_models = [
                "qwen3-embedding",                  # Best quality (8B params)
                "nomic-embed-text-v2-moe",          # Strong MoE model
                "mxbai-embed-large",                 # Good general purpose
                "snowflake-arctic-embed",            # Strong retrieval
                "bge-m3", "bge-large",               # BAAI models
                "nomic-embed-text",                  # Solid default
                "all-minilm",                        # Smallest/fastest
            ]
            available_embed = None
            for em in embedding_models:
                if em in model_names:
                    available_embed = em
                    break
            if available_embed:
                logger.info("Using embedding model: %s", available_embed)
                return OllamaEmbeddings(model=available_embed)
            else:
                logger.debug(
                    "Ollama running but no embedding model found. "
                    "Pull one with: ollama pull nomic-embed-text"
                )
    except Exception:
        pass

    logger.debug("No embedding provider available — RAG disabled")
    return None
