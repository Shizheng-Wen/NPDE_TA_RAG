"""
Main entry point for running the NPDE TA RAG system as a module.

Usage:
    python -m src [COMMAND]
    
Commands:
    index   - Index all course documents
    ask     - Ask a question
    search  - Search for content
    chat    - Interactive chat mode
    stats   - Show index statistics
"""

from .cli import main

if __name__ == "__main__":
    main()
