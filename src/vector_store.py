import chromadb

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

from llama_index.embeddings.google_genai import (
    GoogleGenAIEmbedding,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from src.config import (
    GOOGLE_API_KEY,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def create_embedding_model():

    embedding_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
    )

    print("Embedding model loaded.")

    return embedding_model


def create_chroma_store():

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


def get_existing_documents(collection):

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


def process_documents(
    nodes,
    collection,
):

    existing_documents = (
        get_existing_documents(
            collection
        )
    )

    # Group incoming nodes by document
    documents = {}

    for node in nodes:

        document_key = node.metadata.get(
            "document_key"
        )

        documents.setdefault(
            document_key,
            []
        ).append(node)

    nodes_to_index = []

    for document_key, document_nodes in (
        documents.items()
    ):

        new_hash = document_nodes[0].metadata.get(
            "content_hash"
        )

        old_hash = existing_documents.get(
            document_key
        )

        # --------------------------------------------
        # New document
        # --------------------------------------------

        if old_hash is None:

            print(
                f"[NEW] {document_key}"
            )

            nodes_to_index.extend(
                document_nodes
            )

        # --------------------------------------------
        # Existing unchanged document
        # --------------------------------------------

        elif old_hash == new_hash:

            print(
                f"[UNCHANGED] {document_key}"
            )

        # --------------------------------------------
        # Existing modified document
        # --------------------------------------------

        else:

            print(
                f"[MODIFIED] {document_key}"
            )

            print(
                "Removing old chunks..."
            )

            collection.delete(
                where={
                    "document_key": document_key
                }
            )

            nodes_to_index.extend(
                document_nodes
            )

    return nodes_to_index


def create_index(nodes):

    embedding_model = create_embedding_model()

    vector_store, collection = (
        create_chroma_store()
    )

    nodes_to_index = process_documents(
        nodes,
        collection,
    )

    print(
        f"Chunks to index: "
        f"{len(nodes_to_index)}"
    )

    storage_context = (
        StorageContext.from_defaults(
            vector_store=vector_store
        )
    )

    # --------------------------------------------
    # Index new/modified documents
    # --------------------------------------------

    if nodes_to_index:

        index = VectorStoreIndex(
            nodes_to_index,
            storage_context=storage_context,
            embed_model=embedding_model,
        )

        print(
            "New/modified documents embedded "
            "and stored."
        )

    # --------------------------------------------
    # No changes
    # --------------------------------------------

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