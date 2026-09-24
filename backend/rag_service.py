from src.vector_store import create_index
from src.ingestion import ingest_documents
from src.rag import create_reranker
from src.generator import create_llm


def load_rag_system():
    print("Loading RAG system...")

    nodes = ingest_documents()

    index = create_index(nodes)

    reranker = create_reranker()

    llm = create_llm()

    print("RAG system loaded.")

    return index, reranker, llm