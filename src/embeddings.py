"""
Embedding generation module using OpenAI API.

Provides functions for generating embeddings for text chunks and queries.
"""

from typing import Sequence

from openai import OpenAI
from tqdm import tqdm

from .config import OPENAI_API_KEY, EMBEDDING_MODEL, validate_config


class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI's embedding models.
    """
    
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 100,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed in one API call
        """
        validate_config()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.batch_size = batch_size
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for batch in iterator:
            response = self.client.embeddings.create(
                model=self.model,
                input=list(batch),
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        
        This is the same as embed_text but semantically named for clarity.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self.embed_text(query)


# Global instance for convenience
_embedding_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def embed_query(query: str) -> list[float]:
    """Convenience function to embed a query."""
    return get_embedding_generator().embed_query(query)


def embed_texts(texts: Sequence[str], show_progress: bool = True) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_generator().embed_texts(texts, show_progress)
