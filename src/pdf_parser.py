"""
PDF parsing module for extracting text from course materials.

Handles mathematical content and maintains document structure.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

import fitz  # PyMuPDF
from tqdm import tqdm

from .config import LECTURES_DIR, HOMEWORK_DIR, EXAMS_DIR, DOC_TYPE_MAP


@dataclass
class ParsedPage:
    """Represents a parsed page from a PDF document."""
    
    text: str
    page_number: int
    source_file: str
    doc_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    
    pages: list[ParsedPage]
    source_file: str
    doc_type: str
    total_pages: int
    
    def get_full_text(self) -> str:
        """Get the concatenated text of all pages."""
        return "\n\n".join(page.text for page in self.pages)


def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving mathematical content.
    
    Args:
        text: Raw extracted text from PDF
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines at start and end
    text = text.strip()
    
    return text


def extract_year_from_filename(filename: str) -> int | None:
    """
    Extract year from exam filename.
    
    Args:
        filename: Name of the exam file
        
    Returns:
        Year as integer or None if not found
    """
    # Match patterns like "2018", "2019", "2020", etc.
    # Also match "21", "22", "23", "24", "25" which represent 2021-2025
    
    # First try 4-digit year
    match = re.search(r'20(1[89]|2[0-5])', filename)
    if match:
        return int(match.group(0))
    
    # Try 2-digit year (21, 22, 23, 24, 25)
    match = re.search(r'NPDE(2[1-5])_', filename)
    if match:
        return 2000 + int(match.group(1))
    
    return None


def extract_exam_type(filename: str) -> str | None:
    """
    Extract exam type (midterm, endterm, summer, winter) from filename.
    
    Args:
        filename: Name of the exam file
        
    Returns:
        Exam type string or None
    """
    filename_lower = filename.lower()
    
    if 'midterm' in filename_lower:
        return 'midterm'
    elif 'endterm' in filename_lower:
        return 'endterm'
    elif 'summer' in filename_lower:
        return 'summer_exam'
    elif 'winter' in filename_lower:
        return 'winter_exam'
    
    return 'exam'


def get_doc_type(file_path: Path) -> str:
    """
    Determine document type based on file path.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Document type string
    """
    parent_name = file_path.parent.name
    return DOC_TYPE_MAP.get(parent_name, "unknown")


def parse_pdf(file_path: Path, show_progress: bool = False) -> ParsedDocument:
    """
    Parse a PDF file and extract text with metadata.
    
    Args:
        file_path: Path to the PDF file
        show_progress: Whether to show progress bar
        
    Returns:
        ParsedDocument with all extracted content
    """
    doc = fitz.open(file_path)
    doc_type = get_doc_type(file_path)
    
    pages = []
    page_iter = tqdm(doc, desc=f"Parsing {file_path.name}") if show_progress else doc
    
    for page_num, page in enumerate(page_iter):
        # Extract text with better formatting preservation
        text = page.get_text("text")
        text = clean_text(text)
        
        # Build page metadata
        metadata = {
            "page": page_num + 1,
            "total_pages": len(doc),
        }
        
        # Add exam-specific metadata
        if doc_type == "exam":
            year = extract_year_from_filename(file_path.name)
            exam_type = extract_exam_type(file_path.name)
            if year:
                metadata["year"] = year
            if exam_type:
                metadata["exam_type"] = exam_type
        
        parsed_page = ParsedPage(
            text=text,
            page_number=page_num + 1,
            source_file=str(file_path),
            doc_type=doc_type,
            metadata=metadata,
        )
        pages.append(parsed_page)
    
    doc.close()
    
    return ParsedDocument(
        pages=pages,
        source_file=str(file_path),
        doc_type=doc_type,
        total_pages=len(pages),
    )


def iter_all_pdfs() -> Iterator[Path]:
    """
    Iterate over all PDF files in the data directories.
    
    Yields:
        Path objects for each PDF file
    """
    for directory in [LECTURES_DIR, HOMEWORK_DIR, EXAMS_DIR]:
        if directory.exists():
            for pdf_file in sorted(directory.glob("*.pdf")):
                yield pdf_file


def parse_all_documents(show_progress: bool = True) -> list[ParsedDocument]:
    """
    Parse all PDF documents in the data directories.
    
    Args:
        show_progress: Whether to show progress bars
        
    Returns:
        List of ParsedDocument objects
    """
    documents = []
    pdf_files = list(iter_all_pdfs())
    
    print(f"Found {len(pdf_files)} PDF files to parse")
    
    for pdf_path in pdf_files:
        try:
            doc = parse_pdf(pdf_path, show_progress=show_progress)
            documents.append(doc)
            print(f"  ✓ Parsed {pdf_path.name}: {doc.total_pages} pages")
        except Exception as e:
            print(f"  ✗ Error parsing {pdf_path.name}: {e}")
    
    return documents


if __name__ == "__main__":
    # Test the parser
    docs = parse_all_documents()
    print(f"\nTotal documents parsed: {len(docs)}")
    for doc in docs:
        print(f"  - {Path(doc.source_file).name}: {doc.total_pages} pages ({doc.doc_type})")
