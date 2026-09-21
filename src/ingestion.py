import hashlib
import re
from pathlib import Path
from src.document_classifier import classify_document
from pypdf import PdfReader

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)



from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ============================================================
# 1. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted document text while preserving
    meaningful content and paragraph structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces at beginning/end of lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()


# ============================================================
# 2. COMPANY ID
# ============================================================

def extract_company_id(file_path):
    """
    Determine company ID from the directory structure.

    Example:

    data/google/file.pdf
        -> google

    data/novatech/file.txt
        -> novatech
    """

    path = Path(file_path)
    parts = path.parts

    try:
        data_index = parts.index("data")
    except ValueError:
        return "unknown"

    # Company directory immediately after data/
    if data_index + 1 < len(parts):
        return parts[data_index + 1].lower()

    return "unknown"


# ============================================================
# 3. DOCUMENT METADATA
# ============================================================

def create_document_metadata(
    file_path,
    content,
    content_hash=None,
):
    """
    Create stable metadata for each document.

    content_hash can be supplied externally so that
    all pages of the same PDF share one document-level hash.
    """

    path = Path(file_path)

    company_id = extract_company_id(
        file_path
    )

    document_name = path.name

    # --------------------------------------------------------
    # Document ID
    # --------------------------------------------------------

    document_id = document_name

    # --------------------------------------------------------
    # Content hash
    # --------------------------------------------------------

    # For TXT files, calculate the hash from the
    # complete document content.

    # For PDFs, load_pdf_file() provides a document-level
    # hash calculated from the complete PDF file.

    if content_hash is None:
        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------------
    # File type
    # --------------------------------------------------------

    extension = path.suffix.lower()

    if extension == ".pdf":
        file_type = "pdf"

    elif extension == ".txt":
        file_type = "txt"

    else:
        file_type = extension.replace(
            ".",
            ""
        )

    # --------------------------------------------------------
    # Document type
    # --------------------------------------------------------
    document_type = classify_document(
    file_path=file_path,
    text=content,
)
    
    # --------------------------------------------------------
    # Source type
    # --------------------------------------------------------

    if company_id == "google":
        source_type = "public"

    elif company_id == "novatech":
        source_type = "synthetic_demo"

    else:
        source_type = "unknown"

    # --------------------------------------------------------
    # Final metadata
    # --------------------------------------------------------

    metadata = {
        "company_id": company_id,
        "document_id": document_id,
        "document_key": f"{company_id}/{document_id}",
        "document_name": document_name,
        "source_file": document_name,
        "file_type": file_type,
        "content_hash": content_hash,
        "document_type": document_type,
        "source_type": source_type,
    }

    return metadata


# ============================================================
# 4. LOAD TXT FILE
# ============================================================

def load_txt_file(file_path):
    """
    Load a TXT document.
    """

    path = Path(file_path)

    try:
        text = path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        text = path.read_text(
            encoding="latin-1"
        )

    # Clean text
    text = clean_text(text)

    # Create metadata
    metadata = create_document_metadata(
        file_path,
        text,
    )

    # Create LlamaIndex Document
    document = Document(
        text=text,
        metadata=metadata,
    )

    return document


# ============================================================
# 5. LOAD PDF FILE
# ============================================================

def load_pdf_file(file_path):
    """
    Load a PDF using pypdf.

    Each PDF page becomes a separate Document
    so that page-level citations can be preserved.

    IMPORTANT:
    All pages belonging to the same PDF receive
    the SAME document-level content hash.
    """

    path = Path(file_path)

    reader = PdfReader(
        str(path)
    )

    # --------------------------------------------------------
    # Create ONE hash for the entire PDF
    # --------------------------------------------------------

    file_bytes = path.read_bytes()

    document_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    documents = []

    # --------------------------------------------------------
    # Process each page
    # --------------------------------------------------------

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        # Extract page text
        text = page.extract_text()

        # Clean page text
        text = clean_text(
            text or ""
        )

        # Skip completely empty pages
        if not text:
            continue

        # ----------------------------------------------------
        # Create metadata
        # ----------------------------------------------------

        # IMPORTANT:
        # Pass the SAME document_hash to every page.
        metadata = create_document_metadata(
            file_path,
            text,
            content_hash=document_hash,
        )

        # Page-level metadata
        metadata["page_number"] = str(
            page_number
        )

        # ----------------------------------------------------
        # Create Document
        # ----------------------------------------------------

        document = Document(
            text=text,
            metadata=metadata,
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# 6. LOAD ALL DOCUMENTS
# ============================================================

def load_documents():
    """
    Recursively load TXT and PDF files
    from the data directory.

    PDF:
        pypdf extraction

    TXT:
        UTF-8 / Latin-1 extraction
    """

    data_path = Path(
        DATA_DIR
    )

    documents = []

    # --------------------------------------------------------
    # TXT files
    # --------------------------------------------------------

    for file_path in sorted(
        data_path.rglob("*.txt")
    ):

        document = load_txt_file(
            file_path
        )

        documents.append(
            document
        )

    # --------------------------------------------------------
    # PDF files
    # --------------------------------------------------------

    for file_path in sorted(
        data_path.rglob("*.pdf")
    ):

        pdf_documents = load_pdf_file(
            file_path
        )

        documents.extend(
            pdf_documents
        )

    print(
        f"Document pages/units loaded: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# 7. CREATE CHUNKS
# ============================================================

def create_chunks(documents):
    """
    Split documents into semantically meaningful chunks.

    SentenceSplitter works with token-based chunk sizes.

    Metadata from the original document is preserved.
    """

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_nodes = []

    for document in documents:

        nodes = splitter.get_nodes_from_documents(
            [document]
        )

        for node in nodes:

            # Preserve document metadata
            node.metadata.update(
                document.metadata
            )

        all_nodes.extend(
            nodes
        )

    print(
        f"Chunks created: "
        f"{len(all_nodes)}"
    )

    return all_nodes


# ============================================================
# 8. FULL INGESTION PIPELINE
# ============================================================

def ingest_documents():
    """
    Complete ingestion pipeline:

    Files
      ↓
    Extraction
      ↓
    Cleaning
      ↓
    Metadata
      ↓
    Chunking
      ↓
    Nodes
    """

    documents = load_documents()

    nodes = create_chunks(
        documents
    )

    return nodes