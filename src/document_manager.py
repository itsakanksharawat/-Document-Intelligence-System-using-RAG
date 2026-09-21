from pathlib import Path

import chromadb

from src.config import (
    DATA_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
)


def list_documents(company_id):
    """
    Return documents belonging to a company
    along with their metadata.
    """

    company_id = company_id.strip().lower()

    if not company_id:
        raise ValueError(
            "Company ID cannot be empty."
        )

    company_dir = (
        Path(DATA_DIR) / company_id
    )

    if not company_dir.exists():
        return []

    # Load document metadata through ingestion
    from src.ingestion import load_documents

    all_documents = load_documents()

    documents = []

    seen = set()

    for document in all_documents:

        metadata = document.metadata

        if metadata.get("company_id") != company_id:
            continue

        document_name = metadata.get(
            "document_name"
        )

        if document_name in seen:
            continue

        seen.add(document_name)

        documents.append({
            "document_name": document_name,
            "file_type": metadata.get(
                "file_type",
                "unknown"
            ),
            "document_type": metadata.get(
                "document_type",
                "unknown"
            ),
            "source_type": metadata.get(
                "source_type",
                "unknown"
            ),
            "path": str(
                company_dir / document_name
            ),
        })

    return documents
    

def get_document_metadata(
    company_id,
    document_name,
):
    """
    Get metadata for a specific document.
    """

    company_id = company_id.strip().lower()
    document_name = document_name.strip()

    if not company_id:
        raise ValueError(
            "Company ID cannot be empty."
        )

    if not document_name:
        raise ValueError(
            "Document name cannot be empty."
        )

    from src.ingestion import (
        ingest_documents,
    )

    nodes = ingest_documents()

    for node in nodes:
        metadata = node.metadata

        if (
            metadata.get("company_id")
            == company_id
            and metadata.get("document_name")
            == document_name
        ):
            return {
                "company_id": metadata.get(
                    "company_id"
                ),
                "document_name": metadata.get(
                    "document_name"
                ),
                "document_type": metadata.get(
                    "document_type"
                ),
                "file_type": metadata.get(
                    "file_type"
                ),
                "source_type": metadata.get(
                    "source_type"
                ),
                "content_hash": metadata.get(
                    "content_hash"
                ),
            }

    return None

def delete_document(
    company_id,
    document_name,
):
    """
    Delete a document from both:

    1. The data directory
    2. ChromaDB
    """

    company_id = company_id.strip().lower()
    document_name = document_name.strip()

    if not company_id:
        raise ValueError(
            "Company ID cannot be empty."
        )

    if not document_name:
        raise ValueError(
            "Document name cannot be empty."
        )

    company_dir = (
        Path(DATA_DIR) / company_id
    )

    file_path = (
        company_dir / document_name
    )

    # -------------------------------------------------
    # CHECK FILE
    # -------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    # -------------------------------------------------
    # DELETE FROM CHROMADB
    # -------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    document_key = (
        f"{company_id}/{document_name}"
    )

    collection.delete(
        where={
            "document_key": document_key
        }
    )

    print(
        f"Removed ChromaDB chunks for: "
        f"{document_key}"
    )

    # -------------------------------------------------
    # DELETE PHYSICAL FILE
    # -------------------------------------------------

    file_path.unlink()

    print(
        f"Deleted document: "
        f"{file_path}"
    )

    return True
from src.upload import upload_document
from src.ingestion import ingest_documents
from src.vector_store import create_index


def update_document(
    file_path,
    company_id,
):
    """
    Replace an existing document with a new version
    and re-index it.
    """

    company_id = company_id.strip().lower()

    # -------------------------------------------------
    # COPY NEW VERSION
    # -------------------------------------------------

    uploaded_path = upload_document(
        file_path=file_path,
        company_id=company_id,
    )

    print("\nRe-indexing updated document...")

    # -------------------------------------------------
    # INGEST ALL DOCUMENTS
    # -------------------------------------------------

    nodes = ingest_documents()

    # -------------------------------------------------
    # EXISTING HASH LOGIC HANDLES UPDATE
    # -------------------------------------------------

    create_index(nodes)

    print("\nDocument updated and re-indexed.")

    return uploaded_path