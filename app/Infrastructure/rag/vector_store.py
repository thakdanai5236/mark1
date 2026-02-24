"""
Vector Store - Manages vector embeddings storage and retrieval
"""

import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np


class VectorStore:
    """Simple vector store implementation using numpy."""
    
    def __init__(self, store_path: str):
        """
        Initialize vector store.
        
        Args:
            store_path: Path to store the vector index
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings: List[np.ndarray] = []
        self.documents: List[Dict[str, Any]] = []
        self._load_index()
    
    def _load_index(self):
        """Load existing index from disk."""
        embeddings_path = self.store_path / "embeddings.npy"
        documents_path = self.store_path / "documents.json"
        
        if embeddings_path.exists() and documents_path.exists():
            self.embeddings = list(np.load(str(embeddings_path)))
            with open(documents_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
    
    def save_index(self):
        """Save index to disk."""
        if self.embeddings:
            np.save(
                str(self.store_path / "embeddings.npy"),
                np.array(self.embeddings)
            )
        
        with open(self.store_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def add_documents(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        sources: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            contents: List of document contents
            embeddings: List of embedding vectors
            sources: Optional list of source identifiers
            metadata: Optional list of metadata dicts
        """
        for i, (content, embedding) in enumerate(zip(contents, embeddings)):
            self.embeddings.append(np.array(embedding))
            doc = {
                "content": content,
                "source": sources[i] if sources else f"doc_{len(self.documents)}",
                "metadata": metadata[i] if metadata else {}
            }
            self.documents.append(doc)
        
        self.save_index()
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Filter by metadata fields
            
        Returns:
            List of documents with similarity scores
        """
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        
        # Calculate cosine similarity
        scores = []
        for i, emb in enumerate(self.embeddings):
            # Apply metadata filter
            if filter_metadata:
                doc_meta = self.documents[i].get("metadata", {})
                if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            similarity = np.dot(query_vec, emb) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb)
            )
            scores.append((i, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for idx, score in scores[:k]:
            result = {
                **self.documents[idx],
                "score": float(score)
            }
            results.append(result)
        
        return results
    
    def delete_by_source(self, source: str):
        """Delete all documents from a specific source."""
        indices_to_remove = [
            i for i, doc in enumerate(self.documents)
            if doc.get("source") == source
        ]
        
        for i in reversed(indices_to_remove):
            self.embeddings.pop(i)
            self.documents.pop(i)
        
        self.save_index()
