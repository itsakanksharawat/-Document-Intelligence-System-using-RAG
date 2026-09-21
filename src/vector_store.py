import chromadb

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
)


# ============================================================
# 1. CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Create the local BGE embedding model.

    The model runs locally.
    No Gemini embedding API is required.
    """

    embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Local embedding model loaded.")

    return embedding_model


# ============================================================
# 2. CREATE CHROMA STORE
# ============================================================

def create_chroma_store():
    """
    Connect to the persistent ChromaDB collection.
    """

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    print("ChromaDB connected.")

    return vector_store, collection


# ============================================================
# 3. GET EXISTING DOCUMENTS
# ============================================================

def get_existing_documents(collection):
    """
    Get existing documents from ChromaDB.

    We use our own `document_key` metadata to identify
    documents instead of Chroma's internal record IDs.
    """

    data = collection.get(
        include=["metadatas"]
    )

    existing_documents = {}

    for metadata in data.get(
        "metadatas",
        []
    ):

        if not metadata:
            continue

        document_key = metadata.get(
            "document_key"
        )

        content_hash = metadata.get(
            "content_hash"
        )

        if document_key and content_hash:

            existing_documents[
                document_key
            ] = content_hash

    return existing_documents


# ============================================================
# 4. GROUP NODES BY DOCUMENT
# ============================================================

def group_nodes_by_document(nodes):
    """
    Group chunks belonging to the same document.

    Example:

    google/Google_Code_of_Conduct.pdf
        -> [chunk1, chunk2, chunk3, ...]

    novatech/NovaTech_Remote_Work_Policy.txt
        -> [chunk1, chunk2, ...]
    """

    documents = {}

    for node in nodes:

        document_key = node.metadata.get(
            "document_key",
            "unknown"
        )

        documents.setdefault(
            document_key,
            []
        ).append(node)

    return documents


# ============================================================
# 5. PROCESS DOCUMENTS
# ============================================================

def process_documents(
    nodes,
    collection,
):
    """
    Determine which documents need indexing.

    Possible states:

        [NEW]
        [UNCHANGED]
        [MODIFIED]
    """

    existing_documents = get_existing_documents(
        collection
    )

    documents = group_nodes_by_document(
        nodes
    )

    nodes_to_index = []

    for (
        document_key,
        document_nodes,
    ) in documents.items():

        first_node = document_nodes[0]

        company_id = first_node.metadata.get(
            "company_id",
            "unknown"
        )

        document_id = first_node.metadata.get(
            "document_id",
            "unknown"
        )

        new_hash = first_node.metadata.get(
            "content_hash"
        )

        old_hash = existing_documents.get(
            document_key
        )

        # ----------------------------------------------------
        # NEW DOCUMENT
        # ----------------------------------------------------

        if old_hash is None:

            print(
                f"[NEW] {document_key}"
            )

            nodes_to_index.extend(
                document_nodes
            )

        # ----------------------------------------------------
        # UNCHANGED DOCUMENT
        # ----------------------------------------------------

        elif old_hash == new_hash:

            print(
                f"[UNCHANGED] {document_key}"
            )

        # ----------------------------------------------------
        # MODIFIED DOCUMENT
        # ----------------------------------------------------

        else:

            print(
                f"[MODIFIED] {document_key}"
            )

            print(
                "Removing old chunks..."
            )

            collection.delete(
                where={
                    "$and": [
                        {
                            "company_id": company_id
                        },
                        {
                            "document_key": document_key
                        },
                    ]
                }
            )

            nodes_to_index.extend(
                document_nodes
            )

    return nodes_to_index


# ============================================================
# 6. CREATE / UPDATE INDEX
# ============================================================

def create_index(nodes):
    """
    Create or update the Chroma-backed vector index.
    """

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = create_embedding_model()

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    vector_store, collection = (
        create_chroma_store()
    )

    # --------------------------------------------------------
    # Determine documents to index
    # --------------------------------------------------------

    nodes_to_index = process_documents(
        nodes,
        collection,
    )

    print(
        f"Chunks to index: "
        f"{len(nodes_to_index)}"
    )

    # --------------------------------------------------------
    # Storage context
    # --------------------------------------------------------

    storage_context = (
        StorageContext.from_defaults(
            vector_store=vector_store
        )
    )

    # --------------------------------------------------------
    # Index new/modified chunks
    # --------------------------------------------------------

    if nodes_to_index:

        index = VectorStoreIndex(
            nodes_to_index,
            storage_context=storage_context,
            embed_model=embedding_model,
        )

        print(
            "New/modified documents embedded "
            "locally and stored."
        )

    else:

        index = VectorStoreIndex(
            [],
            storage_context=storage_context,
            embed_model=embedding_model,
        )

        print(
            "No document changes detected."
        )

    return index