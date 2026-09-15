import os

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# ============================================================
# API
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env"
    )


# ============================================================
# Paths
# ============================================================

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"


# ============================================================
# ChromaDB
# ============================================================
COLLECTION_NAME = "company_knowledge"


# ============================================================
# RAG configuration
# ============================================================

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K = 3


# ============================================================
# Models
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"