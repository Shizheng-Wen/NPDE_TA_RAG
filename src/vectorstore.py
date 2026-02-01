"""
Vector store module using ChromaDB for document storage and retrieval.

Provides persistent storage for document chunks and their embeddings.
"""

# Disable ChromaDB telemetry BEFORE importing chromadb
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from .config import VECTORDB_DIR
from .chunker import TextChunk
from .embeddings import EmbeddingGenerator, embed_texts


class VectorStore:
    """
    Vector store backed by ChromaDB for storing and retrieving document chunks.
    """
    
    COLLECTION_NAME = "npde_documents"
    
    def __init__(self, persist_directory: Path | str = VECTORDB_DIR):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "NPDE course materials"},
        )
    
    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        show_progress: bool = True,
    ) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects to add
            show_progress: Whether to show progress bar
        """
        if not chunks:
            print("No chunks to add.")
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        # Remove 'text' from metadata since it's stored as document
        for meta in metadatas:
            meta.pop("text", None)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embed_texts(documents, show_progress=show_progress)
        
        # Add to collection in batches (ChromaDB has limits)
        batch_size = 500
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"Adding chunks to vector store...")
        for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="Storing"):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        doc_type: str | None = None,
        year: int | None = None,
    ) -> dict:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            doc_type: Optional filter by document type (lecture/homework/exam)
            year: Optional filter by year (for exams)
            
        Returns:
            ChromaDB query results
        """
        # Build where clause for filtering
        where = None
        where_conditions = []
        
        if doc_type:
            where_conditions.append({"doc_type": doc_type})
        if year:
            where_conditions.append({"year": year})
        
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}
        
        # Generate query embedding
        embedding_gen = EmbeddingGenerator()
        query_embedding = embedding_gen.embed_query(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        return results
    
    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # ChromaDB doesn't have a direct "delete all" so we recreate the collection
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "NPDE course materials"},
        )
        print("✓ Deleted all documents from vector store")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.count
        
        # Get unique document types by querying all metadata
        doc_types = set()
        years = set()
        
        if count > 0:
            # Get all documents to collect unique types (in batches if needed)
            batch_size = 1000
            offset = 0
            
            while offset < count:
                # Use get() to fetch documents with offset
                result = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                
                if result["metadatas"]:
                    for meta in result["metadatas"]:
                        if meta and "doc_type" in meta:
                            doc_types.add(meta["doc_type"])
                        if meta and "year" in meta:
                            years.add(meta["year"])
                
                offset += batch_size
        
        return {
            "total_chunks": count,
            "doc_types": sorted(doc_types),
            "years": sorted(years) if years else [],
        }


# Global instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def index_documents(chunks: Sequence[TextChunk], show_progress: bool = True) -> None:
    """
    Index document chunks into the vector store.
    
    Args:
        chunks: List of TextChunk objects
        show_progress: Whether to show progress
    """
    store = get_vector_store()
    store.add_chunks(chunks, show_progress=show_progress)


def search(
    query: str,
    n_results: int = 5,
    doc_type: str | None = None,
    year: int | None = None,
) -> dict:
    """
    Search the vector store.
    
    Args:
        query: Search query
        n_results: Number of results
        doc_type: Optional document type filter
        year: Optional year filter
        
    Returns:
        Search results
    """
    store = get_vector_store()
    return store.query(query, n_results=n_results, doc_type=doc_type, year=year)
