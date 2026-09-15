import hashlib
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():

    documents = SimpleDirectoryReader(
        DATA_DIR,
        recursive=True,
    ).load_data()

    print(
        f"Documents loaded: {len(documents)}"
    )

    return documents


def create_document_metadata(document):

    content = document.text

    document_key = document.metadata.get(
        "file_name",
        "unknown"
    )

    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    # --------------------------------------------------------
    # Determine company from folder
    # --------------------------------------------------------

    file_path = document.metadata.get(
        "file_path",
        ""
    )

    normalized_path = file_path.replace(
        "\\",
        "/"
    )

    data_prefix = "data/"

    if data_prefix in normalized_path:

        relative_path = normalized_path.split(
            data_prefix,
            1
        )[1]

        company_id = relative_path.split(
            "/",
            1
        )[0]

    else:

        company_id = "unknown"

    return (
        document_key,
        content_hash,
        company_id,
    )


def create_chunks(documents):

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_nodes = []

    for document in documents:

        (
            document_key,
            content_hash,
            company_id,
        ) = create_document_metadata(
            document
        )

        nodes = splitter.get_nodes_from_documents(
            [document]
        )

        for node in nodes:

            node.metadata[
                "document_key"
            ] = document_key

            node.metadata[
                "content_hash"
            ] = content_hash

            node.metadata[
                "company_id"
            ] = company_id

            node.metadata[
                "source_file"
            ] = document_key

        all_nodes.extend(nodes)

    print(
        f"Chunks created: {len(all_nodes)}"
    )

    return all_nodes


def ingest_documents():

    documents = load_documents()

    nodes = create_chunks(
        documents
    )

    return nodes