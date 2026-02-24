"""
Context Builder - Constructs context from retrieved documents
"""

from typing import List, Optional
from app.Infrastructure.rag.retriever import RetrievedDocument


class ContextBuilder:
    """Builds context strings from retrieved documents."""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        include_sources: bool = True,
        separator: str = "\n\n---\n\n"
    ):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Maximum approximate tokens for context
            include_sources: Whether to include source references
            separator: Separator between documents
        """
        self.max_tokens = max_tokens
        self.include_sources = include_sources
        self.separator = separator
        # Rough estimate: 1 token ≈ 4 characters
        self.char_limit = max_tokens * 4
    
    def build_context(
        self,
        documents: List[RetrievedDocument],
        min_score: float = 0.0
    ) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            min_score: Minimum relevance score threshold
            
        Returns:
            Formatted context string
        """
        # Filter by minimum score
        filtered_docs = [d for d in documents if d.score >= min_score]
        
        if not filtered_docs:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for doc in filtered_docs:
            # Build document section
            if self.include_sources:
                section = f"[Source: {doc.source}]\n{doc.content}"
            else:
                section = doc.content
            
            # Check if adding this would exceed limit
            section_chars = len(section) + len(self.separator)
            if total_chars + section_chars > self.char_limit:
                # Try to add truncated version
                remaining = self.char_limit - total_chars - len(self.separator)
                if remaining > 200:  # Only add if meaningful content
                    section = section[:remaining] + "..."
                    context_parts.append(section)
                break
            
            context_parts.append(section)
            total_chars += section_chars
        
        return self.separator.join(context_parts)
    
    def build_structured_context(
        self,
        documents: List[RetrievedDocument],
        context_type: str = "general"
    ) -> str:
        """
        Build structured context with headers.
        
        Args:
            documents: List of retrieved documents
            context_type: Type of context for custom formatting
            
        Returns:
            Structured context string
        """
        if not documents:
            return ""
        
        header = f"=== Retrieved {context_type.title()} Information ===\n\n"
        
        sections = []
        for i, doc in enumerate(documents, 1):
            section = f"[{i}] Source: {doc.source} (Relevance: {doc.score:.2f})\n"
            section += doc.content
            sections.append(section)
        
        return header + self.separator.join(sections)
    
    def build_summary_context(
        self,
        documents: List[RetrievedDocument],
        max_docs: int = 3
    ) -> str:
        """
        Build a concise summary context with top documents.
        
        Args:
            documents: List of retrieved documents
            max_docs: Maximum number of documents to include
            
        Returns:
            Summary context string
        """
        top_docs = documents[:max_docs]
        
        if not top_docs:
            return "No relevant information found."
        
        parts = []
        for doc in top_docs:
            # Extract first paragraph or first 500 chars
            content = doc.content
            if "\n\n" in content:
                content = content.split("\n\n")[0]
            if len(content) > 500:
                content = content[:500] + "..."
            
            parts.append(f"• {content}")
        
        return "Key Information:\n" + "\n\n".join(parts)
