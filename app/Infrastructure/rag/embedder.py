"""
Embedder - Handles text embedding generation
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import requests

class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass

class OllamaEmbedder(BaseEmbedder):
    """Ollama-based embedder implementation."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 100
    ):
        """
        Initialize Ollama embedder.

        Args:
            model: Ollama embedding model name (e.g., nomic-embed-text)
            base_url: Ollama server URL
            batch_size: Batch size for embedding multiple documents
        """
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

    def _embed_single(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_query(self, text: str) -> List[float]:
        """
        Behaves exactly like OpenAIEmbedder.embed_query
        """
        return self._embed_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Behaves exactly like OpenAIEmbedder.embed_documents
        - Returns List[List[float]]
        - Preserves order
        - Supports batching
        """

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Ollama embedding API typically handles one prompt at a time reliably
            for text in batch:
                embedding = self._embed_single(text)
                all_embeddings.append(embedding)

        return all_embeddings


class LocalEmbedder(BaseEmbedder):
    """Local embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedder.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy initialization of sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
