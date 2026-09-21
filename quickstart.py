from src.ingestion import ingest_documents
from src.vector_store import create_index
from src.rag import ask_rag, create_reranker
from src.generator import create_llm


# ---------------------------------------------------------
# INGEST DOCUMENTS
# ---------------------------------------------------------

nodes = ingest_documents()


# ---------------------------------------------------------
# CREATE VECTOR INDEX
# ---------------------------------------------------------

index = create_index(nodes)


# ---------------------------------------------------------
# LOAD LLM
# ---------------------------------------------------------

llm = create_llm()


# ---------------------------------------------------------
# LOAD RERANKER
# ---------------------------------------------------------

reranker = create_reranker()


# ---------------------------------------------------------
# GET AVAILABLE COMPANIES
# ---------------------------------------------------------

companies = sorted(
    set(
        node.metadata.get(
            "company_id",
            "unknown"
        )
        for node in nodes
    )
)


print("\n" + "=" * 70)
print("COMPANY KNOWLEDGE COPILOT")
print("=" * 70)


print("\nAvailable companies:")

for i, company in enumerate(
    companies,
    start=1,
):
    print(
        f"{i}. {company}"
    )


# ---------------------------------------------------------
# SELECT COMPANY
# ---------------------------------------------------------

while True:

    choice = input(
        "\nSelect company number: "
    )

    try:

        choice = int(choice)

        if 1 <= choice <= len(companies):

            company_id = companies[
                choice - 1
            ]

            break

        print(
            "Invalid choice. "
            "Please select a valid number."
        )

    except ValueError:

        print(
            "Please enter a number."
        )


# ---------------------------------------------------------
# COMPANY SELECTED
# ---------------------------------------------------------

print("\n" + "=" * 70)

print(
    f"COMPANY SELECTED: "
    f"{company_id.upper()}"
)

print("=" * 70)

print(
    "Ask questions about this company's documents."
)

print(
    "Type 'exit' to quit."
)


# ---------------------------------------------------------
# QUESTION LOOP
# ---------------------------------------------------------

while True:

    question = input(
        "\nYou: "
    )

    if question.lower().strip() == "exit":

        print(
            "Goodbye!"
        )

        break

    if not question.strip():

        print(
            "Please enter a question."
        )

        continue


    # -----------------------------------------------------
    # RUN RAG
    # -----------------------------------------------------

    result = ask_rag(
        question=question,
        index=index,
        company_id=company_id,
        reranker=reranker,
        llm=llm,
    )


    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    print("\nAssistant:")

    print(
        result["answer"]
    )


    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    print("\nSources:")

    if result["sources"]:

        for source in result["sources"]:

            print(
                f"- {source}"
            )

    else:

        print(
            "- No supporting sources found."
        )