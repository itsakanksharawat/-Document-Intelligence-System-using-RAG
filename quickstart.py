from src.ingestion import ingest_documents

from src.vector_store import create_index

from src.rag import (
    create_llm,
    create_retriever,
    ask_rag,
)


# ============================================================
# 1. Ingest documents
# ============================================================

nodes = ingest_documents()


# ============================================================
# 2. Create vector index
# ============================================================

index = create_index(nodes)


# ============================================================
# 3. Create LLM
# ============================================================

llm = create_llm()


# ============================================================
# 4. Create retriever
# ============================================================

retriever = create_retriever(index)


# ============================================================
# 5. Start chatbot
# ============================================================

print("\n" + "=" * 70)
print("NOVATECH COMPANY KNOWLEDGE COPILOT")
print("=" * 70)

print("Ask questions about the company policies.")
print("Type 'exit' to quit.")


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":

        print("Goodbye!")

        break

    response, retrieved_nodes = ask_rag(
        question,
        retriever,
        llm,
    )

    print("\nAssistant:")
    print(response)

    print("\nSources:")

    seen_sources = set()

    for node in retrieved_nodes:

        source = node.metadata.get(
            "file_name",
            "Unknown source",
        )

        if source not in seen_sources:

            print(f"- {source}")

            seen_sources.add(source)