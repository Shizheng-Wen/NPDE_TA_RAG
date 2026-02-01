"""
NPDE TA RAG System

A Retrieval-Augmented Generation system for the Numerical Methods for PDE course.
"""

# Disable ChromaDB telemetry before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

__version__ = "0.1.0"
