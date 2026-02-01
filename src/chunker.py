"""
Text chunking module for splitting documents into retrievable chunks.

Implements semantic-aware chunking strategies for mathematical content.
"""

import re
from dataclasses import dataclass, field
from typing import Iterator

import tiktoken

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .pdf_parser import ParsedDocument, ParsedPage


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata for retrieval."""
    
    text: str
    chunk_id: str
    source_file: str
    doc_type: str
    page_start: int
    page_end: int
    token_count: int
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
            **self.metadata,
        }


class TextChunker:
    """
    Chunker that splits text into overlapping chunks.
    
    Uses tiktoken for accurate token counting compatible with OpenAI models.
    """
    
    # Semantic split patterns for mathematical documents
    SPLIT_PATTERNS = [
        r'\n\s*(?:Chapter|CHAPTER)\s+\d+',  # Chapter headings
        r'\n\s*(?:Section|SECTION)\s+\d+',  # Section headings
        r'\n\s*\d+\.\d+(?:\.\d+)?\s+[A-Z]',  # Numbered subsections (e.g., "3.2.1 Finite")
        r'\n\s*(?:Theorem|THEOREM|Lemma|LEMMA|Proposition|PROPOSITION|Corollary|COROLLARY)\s+\d+',
        r'\n\s*(?:Definition|DEFINITION)\s+\d+',
        r'\n\s*(?:Example|EXAMPLE)\s+\d+',
        r'\n\s*(?:Remark|REMARK)\s+\d+',
        r'\n\s*(?:Problem|PROBLEM)\s+\d+',
        r'\n\s*(?:Exercise|EXERCISE)\s+\d+',
        r'\n\s*(?:Solution|SOLUTION)',
        r'\n\s*\(\s*[a-z]\s*\)',  # Sub-problems like (a), (b), (c)
    ]
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Compile split pattern
        self.split_pattern = re.compile(
            '|'.join(self.SPLIT_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def find_split_points(self, text: str) -> list[int]:
        """
        Find semantic split points in text.
        
        Returns list of character positions where splits can occur.
        """
        split_points = [0]
        
        for match in self.split_pattern.finditer(text):
            split_points.append(match.start())
        
        # Also split on double newlines (paragraph breaks)
        for match in re.finditer(r'\n\n+', text):
            split_points.append(match.start())
        
        split_points.append(len(text))
        return sorted(set(split_points))
    
    def split_into_segments(self, text: str) -> list[str]:
        """
        Split text into semantic segments.
        
        These segments are then combined into chunks of appropriate size.
        """
        split_points = self.find_split_points(text)
        segments = []
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
        
        return segments
    
    def chunk_text(
        self,
        text: str,
        source_file: str,
        doc_type: str,
        base_metadata: dict | None = None,
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            source_file: Source file path
            doc_type: Document type (lecture/homework/exam)
            base_metadata: Additional metadata to include
            
        Returns:
            List of TextChunk objects
        """
        if base_metadata is None:
            base_metadata = {}
        
        segments = self.split_into_segments(text)
        chunks = []
        
        current_chunk_segments = []
        current_token_count = 0
        chunk_index = 0
        
        for segment in segments:
            segment_tokens = self.count_tokens(segment)
            
            # If single segment exceeds chunk size, split it further
            if segment_tokens > self.chunk_size:
                # First, flush current chunk if not empty
                if current_chunk_segments:
                    chunk_text = "\n\n".join(current_chunk_segments)
                    chunk = TextChunk(
                        text=chunk_text,
                        chunk_id=f"{source_file}::chunk_{chunk_index}",
                        source_file=source_file,
                        doc_type=doc_type,
                        page_start=base_metadata.get("page_start", 1),
                        page_end=base_metadata.get("page_end", 1),
                        token_count=self.count_tokens(chunk_text),
                        metadata=base_metadata.copy(),
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_segments = []
                    current_token_count = 0
                
                # Split large segment by sentences or fixed size
                sub_chunks = self._split_large_segment(segment)
                for sub_chunk in sub_chunks:
                    chunk = TextChunk(
                        text=sub_chunk,
                        chunk_id=f"{source_file}::chunk_{chunk_index}",
                        source_file=source_file,
                        doc_type=doc_type,
                        page_start=base_metadata.get("page_start", 1),
                        page_end=base_metadata.get("page_end", 1),
                        token_count=self.count_tokens(sub_chunk),
                        metadata=base_metadata.copy(),
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                continue
            
            # Check if adding this segment would exceed chunk size
            if current_token_count + segment_tokens > self.chunk_size:
                # Create chunk from current segments
                if current_chunk_segments:
                    chunk_text = "\n\n".join(current_chunk_segments)
                    chunk = TextChunk(
                        text=chunk_text,
                        chunk_id=f"{source_file}::chunk_{chunk_index}",
                        source_file=source_file,
                        doc_type=doc_type,
                        page_start=base_metadata.get("page_start", 1),
                        page_end=base_metadata.get("page_end", 1),
                        token_count=self.count_tokens(chunk_text),
                        metadata=base_metadata.copy(),
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Keep overlap segments
                    overlap_segments = []
                    overlap_tokens = 0
                    for seg in reversed(current_chunk_segments):
                        seg_tokens = self.count_tokens(seg)
                        if overlap_tokens + seg_tokens <= self.chunk_overlap:
                            overlap_segments.insert(0, seg)
                            overlap_tokens += seg_tokens
                        else:
                            break
                    
                    current_chunk_segments = overlap_segments
                    current_token_count = overlap_tokens
            
            current_chunk_segments.append(segment)
            current_token_count += segment_tokens
        
        # Don't forget the last chunk
        if current_chunk_segments:
            chunk_text = "\n\n".join(current_chunk_segments)
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{source_file}::chunk_{chunk_index}",
                source_file=source_file,
                doc_type=doc_type,
                page_start=base_metadata.get("page_start", 1),
                page_end=base_metadata.get("page_end", 1),
                token_count=self.count_tokens(chunk_text),
                metadata=base_metadata.copy(),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_segment(self, text: str) -> list[str]:
        """Split a large segment into smaller pieces."""
        # Try splitting by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # If single sentence is too long, split by words
                if sentence_tokens > self.chunk_size:
                    words = sentence.split()
                    word_chunk = []
                    word_tokens = 0
                    for word in words:
                        word_token_count = self.count_tokens(word)
                        if word_tokens + word_token_count > self.chunk_size:
                            if word_chunk:
                                chunks.append(" ".join(word_chunk))
                            word_chunk = [word]
                            word_tokens = word_token_count
                        else:
                            word_chunk.append(word)
                            word_tokens += word_token_count
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    continue
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Chunk an entire parsed document.
        
        Args:
            document: ParsedDocument from pdf_parser
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        # Process pages in groups to maintain context
        # but also include page-level metadata
        full_text = document.get_full_text()
        
        base_metadata = {
            "page_start": 1,
            "page_end": document.total_pages,
        }
        
        # Add exam-specific metadata if available
        if document.pages and document.pages[0].metadata:
            first_page_meta = document.pages[0].metadata
            if "year" in first_page_meta:
                base_metadata["year"] = first_page_meta["year"]
            if "exam_type" in first_page_meta:
                base_metadata["exam_type"] = first_page_meta["exam_type"]
        
        chunks = self.chunk_text(
            text=full_text,
            source_file=document.source_file,
            doc_type=document.doc_type,
            base_metadata=base_metadata,
        )
        
        all_chunks.extend(chunks)
        
        return all_chunks


def chunk_documents(documents: list[ParsedDocument]) -> list[TextChunk]:
    """
    Chunk a list of parsed documents.
    
    Args:
        documents: List of ParsedDocument objects
        
    Returns:
        List of all TextChunk objects
    """
    chunker = TextChunker()
    all_chunks = []
    
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  ✓ Chunked {doc.source_file}: {len(chunks)} chunks")
    
    return all_chunks


if __name__ == "__main__":
    from .pdf_parser import parse_all_documents
    
    # Test chunking
    docs = parse_all_documents(show_progress=False)
    chunks = chunk_documents(docs)
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Show sample chunk
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk:")
        print(f"  ID: {sample.chunk_id}")
        print(f"  Type: {sample.doc_type}")
        print(f"  Tokens: {sample.token_count}")
        print(f"  Text preview: {sample.text[:200]}...")
