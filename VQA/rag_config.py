"""
Stage 7 RAG configuration — RTX 4070 machine.
All RAG-related scripts import from here.
"""
import os
from pathlib import Path

# Corpus documents
RAG_CORPUS_DIR = Path(os.environ.get(
    "STAGE7_RAG_CORPUS_DIR",
    "/home/vjti-comp/Desktop/Final Project Code/VQA/rag_corpus"))

# ChromaDB persistent vector store
CHROMA_DB_DIR = Path(os.environ.get(
    "STAGE7_CHROMA_DIR",
    "/home/vjti-comp/Desktop/Final Project Code/VQA/chroma_db"))

CHROMA_COLLECTION_NAME = "dermatology_kb"
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"

CHUNK_SIZE           = 350
CHUNK_OVERLAP        = 60
MIN_CHUNK_CHARS      = 40
TOP_K                = 3
SIMILARITY_THRESHOLD = 0.25
EMBED_BATCH_SIZE     = 64
