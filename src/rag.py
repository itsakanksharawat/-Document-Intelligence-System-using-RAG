import re

from src.generator import generate_answer

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)

from sentence_transformers import CrossEncoder

from src.config import TOP_K


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "of", "to", "in", "on", "for", "and", "or",
    "what", "which", "how", "many", "does",
    "do", "can", "will", "be", "from", "by", "at",
    "as", "about", "their", "they", "it", "this",
    "that", "company", "policy"
}


def tokenize_text(text):
    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    return {
        token
        for token in tokens
        if token not in STOPWORDS
    }


def calculate_evidence_score(
    question,
    text,
):
    question_tokens = tokenize_text(question)
    text_tokens = tokenize_text(text)

    if not question_tokens:
        return 0.0

    overlap = question_tokens.intersection(
        text_tokens
    )

    return len(overlap) / len(question_tokens)


def create_retriever(
    index,
    company_id,
    top_k=TOP_K,
):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="company_id",
                value=company_id,
            )
        ]
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=filters,
    )

    return retriever


def retrieve_documents(
    retriever,
    question,
):
    return retriever.retrieve(question)


def create_reranker():
    print("Loading reranker model...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("Reranker loaded.")

    return reranker


def rerank_nodes(
    question,
    nodes,
    reranker,
):
    if not nodes:
        return []

    pairs = [
        (
            question,
            node.get_content()
        )
        for node in nodes
    ]

    scores = reranker.predict(pairs)

    reranked_nodes = []

    for node, score in zip(
        nodes,
        scores,
    ):
        reranked_node = NodeWithScore(
            node=node.node,
            score=float(score),
        )

        reranked_nodes.append(
            reranked_node
        )

    reranked_nodes.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return reranked_nodes


def remove_duplicate_chunks(
    nodes,
):
    """
    Remove only identical chunks.

    Multiple different chunks from the same
    document are allowed because they may
    contain different pieces of evidence.
    """

    seen_chunks = set()
    unique_nodes = []

    for node in nodes:

        content = node.get_content().strip()

        if content in seen_chunks:
            continue

        seen_chunks.add(content)
        unique_nodes.append(node)

    return unique_nodes


def check_answerability(
    question,
    reranked_nodes,
):
    """
    Heuristic answerability gate.

    Checks:
    - reranker relevance
    - lexical evidence
    - multiple supporting chunks
    """

    if not reranked_nodes:
        return {
            "answerable": False,
            "reason": "No retrieved evidence.",
            "top_reranker_score": 0.0,
            "top_evidence_score": 0.0,
            "supporting_chunks": 0,
        }

    evidence_items = []

    for node in reranked_nodes:

        reranker_score = float(
            node.score or 0.0
        )

        evidence_score = (
            calculate_evidence_score(
                question,
                node.get_content(),
            )
        )

        evidence_items.append(
            {
                "node": node,
                "reranker_score": reranker_score,
                "evidence_score": evidence_score,
            }
        )

    top_item = evidence_items[0]

    top_reranker_score = (
        top_item["reranker_score"]
    )

    top_evidence_score = (
        top_item["evidence_score"]
    )

    supporting_chunks = sum(
        1
        for item in evidence_items
        if item["evidence_score"] >= 0.30
    )

    strong_direct_evidence = (
        top_evidence_score >= 0.50
        and top_reranker_score >= 0.05
    )

    multiple_support = (
        supporting_chunks >= 2
        and top_evidence_score >= 0.30
    )

    combined_evidence = (
        top_evidence_score >= 0.40
        and top_reranker_score >= 0.01
    )

    answerable = (
        strong_direct_evidence
        or multiple_support
        or combined_evidence
    )

    if answerable:
        reason = (
            "Retrieved evidence contains "
            "sufficient supporting signals."
        )
    else:
        reason = (
            "Retrieved evidence does not contain "
            "sufficient support for the question."
        )

    return {
        "answerable": answerable,
        "reason": reason,
        "top_reranker_score": top_reranker_score,
        "top_evidence_score": top_evidence_score,
        "supporting_chunks": supporting_chunks,
        "evidence_items": evidence_items,
    }


def select_relevant_nodes(
    question,
    reranked_nodes,
):
    answerability = check_answerability(
        question,
        reranked_nodes,
    )

    if not answerability["answerable"]:
        return []

    selected = []

    for node in reranked_nodes:

        evidence_score = (
            calculate_evidence_score(
                question,
                node.get_content(),
            )
        )

        reranker_score = float(
            node.score or 0.0
        )

        # Keep strongly relevant evidence.
        if (
            evidence_score >= 0.30
            or reranker_score >= 0.05
        ):
            selected.append(node)

    # Keep only the strongest evidence when
    # the selected set contains weak sources.
    if selected:
        top_score = max(
            calculate_evidence_score(
                question,
                node.get_content(),
            )
            for node in selected
        )

        selected = [
            node
            for node in selected
            if (
                calculate_evidence_score(
                    question,
                    node.get_content(),
                )
                >= max(0.30, top_score * 0.50)
            )
        ]

    if not selected and reranked_nodes:
        selected.append(
            reranked_nodes[0]
        )

    return selected

def print_retrieval_results(
    question,
    reranked_nodes,
):
    print("\n" + "=" * 80)

    print("QUESTION:")
    print(question)

    print("\nRERANKED RESULTS:")

    for rank, node in enumerate(
        reranked_nodes,
        start=1,
    ):
        evidence_score = (
            calculate_evidence_score(
                question,
                node.get_content(),
            )
        )

        print(f"\nRank {rank}")

        print(
            f"Reranker score: "
            f"{node.score:.6f}"
        )

        print(
            f"Evidence score: "
            f"{evidence_score:.3f}"
        )

        print(
            f"Source: "
            f"{node.metadata.get('source_file')}"
        )

        if node.metadata.get("page_number"):
            print(
                f"Page: "
                f"{node.metadata.get('page_number')}"
            )

        print("Text:")

        print(
            node.get_content()[:500]
        )


def ask_rag(
    question,
    index,
    company_id,
    reranker=None,
    llm=None,
):
    retriever = create_retriever(
        index=index,
        company_id=company_id,
    )

    nodes = retrieve_documents(
        retriever,
        question,
    )

    if reranker is None:
        reranker = create_reranker()

    reranked_nodes = rerank_nodes(
        question,
        nodes,
        reranker,
    )

    reranked_nodes = remove_duplicate_chunks(
        reranked_nodes
    )

    selected_nodes = select_relevant_nodes(
        question,
        reranked_nodes,
    )

    if not selected_nodes:
        return {
            "question": question,
            "company_id": company_id,
            "retrieved_nodes": reranked_nodes,
            "selected_nodes": [],
            "answerable": False,
            "answer": (
                "I don't have enough information "
                "in the provided documents to "
                "answer this."
            ),
            "sources": [],
        }

    generated = generate_answer(
        question=question,
        nodes=selected_nodes,
        llm=llm,
    )

    return {
        "question": question,
        "company_id": company_id,
        "retrieved_nodes": reranked_nodes,
        "selected_nodes": selected_nodes,
        "answerable": True,
        "answer": generated["answer"],
        "sources": generated["sources"],
    }