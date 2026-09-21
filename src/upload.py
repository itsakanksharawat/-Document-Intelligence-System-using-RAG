from pathlib import Path
import shutil

from src.config import DATA_DIR
from src.ingestion import ingest_documents
from src.vector_store import create_index


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def upload_document(file_path, company_id, overwrite=False):
    company_id = company_id.strip().lower()

    if not company_id:
        raise ValueError("Company ID cannot be empty.")

    source_path = Path(file_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = source_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Only PDF and TXT files are supported."
        )

    company_dir = Path(DATA_DIR) / company_id
    company_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = company_dir / source_path.name

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Document already exists: {destination}"
        )

    shutil.copy2(
        source_path,
        destination
    )

    print("Document uploaded successfully:")
    print(destination)

    return destination


def upload_and_index_document(
    file_path,
    company_id,
    overwrite=False
):
    uploaded_path = upload_document(
        file_path=file_path,
        company_id=company_id,
        overwrite=overwrite,
    )

    print("\nStarting indexing...")

    nodes = ingest_documents()

    create_index(nodes)

    print("\nDocument is now searchable.")

    return uploaded_path