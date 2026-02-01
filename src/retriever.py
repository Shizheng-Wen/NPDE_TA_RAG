"""
Retriever module for finding relevant document chunks.

Provides high-level retrieval functions with formatting.
"""

from dataclasses import dataclass
from pathlib import Path

from .vectorstore import search, get_vector_store
from .config import TOP_K_RESULTS


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata and relevance score."""
    
    text: str
    source_file: str
    doc_type: str
    page_start: int
    page_end: int
    distance: float
    metadata: dict
    
    @property
    def source_name(self) -> str:
        """Get a readable source name."""
        return Path(self.source_file).name
    
    @property
    def relevance_score(self) -> float:
        """Convert distance to relevance score (0-1, higher is better)."""
        # ChromaDB returns L2 distance, convert to similarity
        return 1 / (1 + self.distance)
    
    def format_source(self) -> str:
        """Format the source for display."""
        source = f"{self.source_name}"
        if self.page_start == self.page_end:
            source += f" (p.{self.page_start})"
        else:
            source += f" (p.{self.page_start}-{self.page_end})"
        return source


class Retriever:
    """
    High-level retriever for finding relevant documents.
    """
    
    def __init__(self, top_k: int = TOP_K_RESULTS):
        """
        Initialize the retriever.
        
        Args:
            top_k: Default number of results to return
        """
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
        year: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides default)
            doc_type: Filter by document type
            year: Filter by year (for exams)
            
        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        k = top_k or self.top_k
        
        results = search(
            query=query,
            n_results=k,
            doc_type=doc_type,
            year=year,
        )
        
        chunks = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                chunk = RetrievedChunk(
                    text=doc,
                    source_file=metadata.get("source_file", "unknown"),
                    doc_type=metadata.get("doc_type", "unknown"),
                    page_start=metadata.get("page_start", 1),
                    page_end=metadata.get("page_end", 1),
                    distance=distance,
                    metadata=metadata,
                )
                chunks.append(chunk)
        
        return chunks
    
    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as context for the LLM.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}: {chunk.format_source()} - {chunk.doc_type}]"
            context_parts.append(f"{header}\n{chunk.text}")
        
        return "\n\n---\n\n".join(context_parts)


# Global instance
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
    year: int | None = None,
) -> list[RetrievedChunk]:
    """Convenience function to retrieve chunks."""
    return get_retriever().retrieve(query, top_k, doc_type, year)


def get_context(
    query: str,
    top_k: int | None = None,
    doc_type: str | None = None,
    year: int | None = None,
) -> str:
    """Get formatted context for a query."""
    retriever = get_retriever()
    chunks = retriever.retrieve(query, top_k, doc_type, year)
    return retriever.format_context(chunks)
