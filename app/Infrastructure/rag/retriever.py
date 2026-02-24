"""
RAG Retriever - Handles document retrieval from vector store
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata."""
    content: str
    source: str
    score: float
    metadata: Optional[dict] = None


class Retriever:
    """Retrieves relevant documents from vector store."""
    
    def __init__(self, vector_store, embedder, top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance for similarity search
            embedder: Embedder instance for query embedding
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Override default top_k
            filter_metadata: Filter results by metadata
            
        Returns:
            List of retrieved documents sorted by relevance
        """
        k = top_k or self.top_k
        
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata
        )
        
        # Convert to RetrievedDocument objects
        documents = []
        for result in results:
            doc = RetrievedDocument(
                content=result["content"],
                source=result.get("source", "unknown"),
                score=result["score"],
                metadata=result.get("metadata")
            )
            documents.append(doc)
        
        return documents
    
    def retrieve_with_rerank(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Retrieve with two-stage retrieval and reranking.
        
        Args:
            query: Search query
            initial_k: Number of candidates for initial retrieval
            final_k: Final number of documents after reranking
            
        Returns:
            Reranked list of retrieved documents
        """
        # Initial retrieval
        candidates = self.retrieve(query, top_k=initial_k)
        
        # TODO: Implement reranking logic
        # For now, just return top final_k
        return candidates[:final_k]
