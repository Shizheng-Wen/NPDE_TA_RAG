"""
Configuration management for the NPDE TA RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LECTURES_DIR = DATA_DIR / "lectures"
HOMEWORK_DIR = DATA_DIR / "homework"
EXAMS_DIR = DATA_DIR / "exams"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"

# Ensure vectordb directory exists
VECTORDB_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Chunking Configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of chunks to retrieve

# Document type mapping
DOC_TYPE_MAP = {
    "lectures": "lecture",
    "homework": "homework",
    "exams": "exam",
}


def validate_config() -> bool:
    """Validate that all required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please create a .env file with your API key. "
            "See .env.example for reference."
        )
    return True
