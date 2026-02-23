"""
Embedder - Handles text embedding generation
"""

from typing import List, Optional
from abc import ABC, abstractmethod


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


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-based embedder implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            batch_size: Batch size for embedding multiple documents
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
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
