from llama_index.llms.google_genai import GoogleGenAI

from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    TOP_K,
)


def create_llm():

    llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
    )

    print("LLM loaded.")

    return llm


def create_retriever(index):

    retriever = index.as_retriever(
        similarity_top_k=TOP_K
    )

    print("Retriever created.")

    return retriever


def ask_rag(
    question,
    retriever,
    llm,
):

    retrieved_nodes = retriever.retrieve(
        question
    )

    context_parts = []

    for node in retrieved_nodes:

        source = node.metadata.get(
            "file_name",
            "Unknown source",
        )

        context_parts.append(
            f"""
SOURCE: {source}

CONTENT:
{node.text}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are NovaTech's internal company
knowledge assistant.

Answer the user's question ONLY using
the provided company documents.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make assumptions.
4. If the answer is not present in the
   documents, say:

"I don't have enough information in the
provided company documents."

5. Give a concise answer.
6. Do not create source numbers.
7. Do not mention SOURCE 1, SOURCE 2, etc.

COMPANY DOCUMENTS:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = llm.complete(prompt)

    return response, retrieved_nodes