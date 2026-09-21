import os

import streamlit as st

from src.rag import (
    ask_rag,
    create_reranker,
)

from src.generator import create_llm

from src.ingestion import ingest_documents

from src.vector_store import create_index

from src.upload import (
    upload_and_index_document,
)

from src.document_manager import (
    list_documents,
    delete_document,
)


# =========================================================
# CACHED RAG SYSTEM
# =========================================================

@st.cache_resource
def load_rag_system():
    """
    Load the RAG components once and cache them.

    This prevents the embedding model,
    ChromaDB index,
    reranker, and Gemini LLM
    from being recreated for every question.
    """

    nodes = ingest_documents()

    index = create_index(
        nodes
    )

    reranker = create_reranker()

    llm = create_llm()

    return index, reranker, llm


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Company Knowledge Copilot",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "📚 Company Knowledge Copilot"
)

st.write(
    "Upload, manage, and query company documents."
)


# =========================================================
# COMPANY
# =========================================================

company_id = st.text_input(
    "Company ID",
    placeholder="e.g. google, novatech",
)


if not company_id.strip():

    st.info(
        "Enter a company ID to continue."
    )

    st.stop()


company_id = (
    company_id
    .strip()
    .lower()
)


# =========================================================
# UPLOAD
# =========================================================

st.header(
    "Upload Document"
)

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=[
        "pdf",
        "txt",
    ],
)


if uploaded_file is not None:

    if st.button(
        "Upload and Index",
        type="primary",
    ):

        # -------------------------------------------------
        # Temporary upload file
        # -------------------------------------------------

        temp_path = (
            f"temp_{uploaded_file.name}"
        )

        with open(
            temp_path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        try:

            with st.spinner(
                "Uploading and indexing..."
            ):

                upload_and_index_document(
                    file_path=temp_path,
                    company_id=company_id,
                )

            st.success(
                "Document uploaded and indexed successfully."
            )
            

        except Exception as error:

            st.error(
                f"Upload failed: {error}"
            )

        finally:

            if os.path.exists(
                temp_path
            ):

                os.remove(
                    temp_path
                )


# =========================================================
# DOCUMENT LIST
# =========================================================

st.header(
    "Company Documents"
)

documents = list_documents(
    company_id
)


if not documents:

    st.info(
        "No documents uploaded for this company."
    )

else:

    for document in documents:

        col1, col2 = st.columns(
            [4, 1]
        )

        # -------------------------------------------------
        # DOCUMENT INFORMATION
        # -------------------------------------------------

        with col1:

          st.write(
        f"📄 {document['document_name']}"
       )

          st.caption(
         f"File type: {document['file_type']} | "
         f"Document type: {document['document_type']} | "
         f"Source: {document['source_type']}"
     )

        # -------------------------------------------------
        # DELETE
        # -------------------------------------------------

        with col2:

            delete_key = (
                f"delete_{company_id}_"
                f"{document['document_name']}"
            )

            if st.button(
                "Delete",
                key=delete_key,
            ):

                try:

                    delete_document(
                        company_id=company_id,
                        document_name=(
                            document[
                                "document_name"
                            ]
                        ),
                    )

                    st.success(
                        "Document deleted."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Delete failed: {error}"
                    )


# =========================================================
# QUESTION ANSWERING
# =========================================================

st.header(
    "Ask Your Documents"
)

question = st.text_input(
    "Ask a question",
    placeholder=(
        "e.g. How many casual leaves are provided?"
    ),
)


if st.button(
    "Ask"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching documents..."
            ):

                # -------------------------------------------------
                # LOAD CACHED RAG SYSTEM
                # -------------------------------------------------

                index, reranker, llm = (
                    load_rag_system()
                )

                # -------------------------------------------------
                # RUN RAG
                # -------------------------------------------------

                result = ask_rag(
                    question=question,
                    index=index,
                    company_id=company_id,
                    reranker=reranker,
                    llm=llm,
                )

            # -------------------------------------------------
            # ANSWER
            # -------------------------------------------------

            st.subheader(
                "Answer"
            )

            st.write(
                result["answer"]
            )

            # -------------------------------------------------
            # SOURCES
            # -------------------------------------------------

            st.subheader(
                "Sources"
            )

            if result["sources"]:

                for source in result[
                    "sources"
                ]:

                    st.write(
                        f"📄 {source}"
                    )

            else:

                st.write(
                    "No supporting sources found."
                )

        except Exception as error:

            st.error(
                f"Question failed: {error}"
            )