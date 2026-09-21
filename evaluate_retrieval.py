from src.ingestion import ingest_documents
from src.vector_store import create_index

from src.rag import (
    create_retriever,
    create_reranker,
    rerank_nodes,
    remove_duplicate_chunks,
    select_relevant_nodes,
)


# =========================================================
# EVALUATION DATASET
# =========================================================

TEST_CASES = [

    # =====================================================
    # DIRECT QUESTIONS
    # =====================================================

    {
        "id": 1,
        "category": "direct",
        "question": "What are the password requirements?",
        "expected_documents": [
            "NovaTech_IT_Security_Policy.txt"
        ],
    },

    {
        "id": 2,
        "category": "direct",
        "question": "How much annual leave do employees get?",
        "expected_documents": [
            "NovaTech_Leave_Policy.txt"
        ],
    },

    {
        "id": 3,
        "category": "direct",
        "question": "What is the onboarding process?",
        "expected_documents": [
            "NovaTech_Onboarding_Guide.txt"
        ],
    },

    {
        "id": 4,
        "category": "direct",
        "question": "How are employee expenses reimbursed?",
        "expected_documents": [
            "NovaTech_Expense_Policy.txt"
        ],
    },


    # =====================================================
    # PARAPHRASED QUESTIONS
    # =====================================================

    {
        "id": 5,
        "category": "paraphrased",
        "question": "Can employees work from home at NovaTech?",
        "expected_documents": [
            "NovaTech_Remote_Work_Policy.txt"
        ],
    },

    {
        "id": 6,
        "category": "paraphrased",
        "question": "What rules apply to employee passwords?",
        "expected_documents": [
            "NovaTech_IT_Security_Policy.txt"
        ],
    },

    {
        "id": 7,
        "category": "paraphrased",
        "question": "What happens when a new employee joins NovaTech?",
        "expected_documents": [
            "NovaTech_Onboarding_Guide.txt"
        ],
    },

    {
        "id": 8,
        "category": "paraphrased",
        "question": "What is the procedure for claiming company expenses?",
        "expected_documents": [
            "NovaTech_Expense_Policy.txt"
        ],
    },


    # =====================================================
    # MULTI-DOCUMENT QUESTIONS
    # =====================================================

    {
        "id": 9,
        "category": "multi_document",
        "question": (
            "Can an employee who has recently joined NovaTech "
            "work from home?"
        ),
        "expected_documents": [
            "NovaTech_Onboarding_Guide.txt",
            "NovaTech_Remote_Work_Policy.txt",
        ],
    },

    {
        "id": 10,
        "category": "multi_document",
        "question": (
            "If I work remotely, can I get internet expenses "
            "reimbursed?"
        ),
        "expected_documents": [
            "NovaTech_Remote_Work_Policy.txt",
            "NovaTech_Expense_Policy.txt",
        ],
    },

    {
        "id": 11,
        "category": "multi_document",
        "question": (
            "What equipment and internet responsibilities "
            "apply to remote employees?"
        ),
        "expected_documents": [
            "NovaTech_Remote_Work_Policy.txt",
            "NovaTech_Expense_Policy.txt",
        ],
    },

    {
        "id": 12,
        "category": "multi_document",
        "question": (
            "What is the probation period and what happens "
            "after probation for remote work eligibility?"
        ),
        "expected_documents": [
            "NovaTech_Onboarding_Guide.txt",
            "NovaTech_Remote_Work_Policy.txt",
        ],
    },


    # =====================================================
    # UNANSWERABLE QUESTIONS
    # =====================================================

    {
        "id": 13,
        "category": "unanswerable",
        "question": (
            "How many days of paid study leave does "
            "NovaTech provide?"
        ),
        "expected_documents": [],
    },

    {
        "id": 14,
        "category": "unanswerable",
        "question": (
            "What is the annual salary increase percentage "
            "at NovaTech?"
        ),
        "expected_documents": [],
    },

    {
        "id": 15,
        "category": "unanswerable",
        "question": (
            "Which health insurance company does "
            "NovaTech use?"
        ),
        "expected_documents": [],
    },

    {
        "id": 16,
        "category": "unanswerable",
        "question": (
            "What is NovaTech's employee bonus structure?"
        ),
        "expected_documents": [],
    },


    # =====================================================
    # COMPANY ISOLATION
    # =====================================================

    {
        "id": 17,
        "category": "company_isolation",
        "question": (
            "What does Google say about employee leave?"
        ),
        "expected_documents": [],
    },

    {
        "id": 18,
        "category": "company_isolation",
        "question": (
            "What is Google's hybrid work policy?"
        ),
        "expected_documents": [],
    },

    {
        "id": 19,
        "category": "company_isolation",
        "question": (
            "What are Google's password requirements?"
        ),
        "expected_documents": [],
    },

    {
        "id": 20,
        "category": "company_isolation",
        "question": (
            "How does Google reimburse employee expenses?"
        ),
        "expected_documents": [],
    },
]


# =========================================================
# DOCUMENT NAME
# =========================================================

def get_document_name(node):

    return node.metadata.get(
        "source_file",
        "Unknown",
    )


# =========================================================
# STANDARD METRICS
# =========================================================

def hit_at_1(
    ranked_documents,
    expected_documents,
):

    if not expected_documents:
        return None

    if not ranked_documents:
        return 0

    return int(
        ranked_documents[0]
        in expected_documents
    )


def recall_at_3(
    ranked_documents,
    expected_documents,
):

    if not expected_documents:
        return None

    retrieved = set(
        ranked_documents[:3]
    )

    expected = set(
        expected_documents
    )

    return (
        len(retrieved & expected)
        / len(expected)
    )


def reciprocal_rank(
    ranked_documents,
    expected_documents,
):

    if not expected_documents:
        return None

    for rank, document in enumerate(
        ranked_documents,
        start=1,
    ):

        if document in expected_documents:

            return 1 / rank

    return 0.0


# =========================================================
# UNANSWERABLE EVALUATION
# =========================================================

def evaluate_unanswerable(
    final_documents,
):
    """
    For an unanswerable question, the retrieval
    pipeline should eventually provide no usable
    supporting context.

    1 = correctly rejected
    0 = supporting context was selected

    NOTE:
    This currently evaluates the context-selection
    behavior. A dedicated answerability gate will
    be added later.
    """

    return int(
        len(final_documents) == 0
    )


# =========================================================
# COMPANY ISOLATION EVALUATION
# =========================================================

def evaluate_company_isolation(
    final_documents,
):
    """
    The active retriever is restricted to NovaTech.

    A Google document appearing in the final context
    would therefore represent company-data leakage.
    """

    leaked_google_documents = [

        document

        for document in final_documents

        if document.lower().startswith(
            "google"
        )
    ]

    return int(
        len(leaked_google_documents) == 0
    )


# =========================================================
# PRINT RANKING
# =========================================================

def print_ranking(
    title,
    ranked_nodes,
    show_reranker_score=False,
):

    print(
        f"\n{title}"
    )

    if not ranked_nodes:

        print(
            "- None"
        )

        return

    for rank, node in enumerate(
        ranked_nodes,
        start=1,
    ):

        document = get_document_name(
            node
        )

        if show_reranker_score:

            score = node.metadata.get(
                "reranker_score",
                0.0,
            )

            evidence = node.metadata.get(
                "evidence_score",
                0.0,
            )

            print(
                f"{rank}. {document} "
                f"(reranker={score:.6f}, "
                f"evidence={evidence:.3f})"
            )

        else:

            print(
                f"{rank}. {document}"
            )


# =========================================================
# AVERAGE
# =========================================================

def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# =========================================================
# MAIN
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "ADVANCED RAG RETRIEVAL EVALUATION"
)

print(
    "=" * 70
)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

nodes = ingest_documents()


# =========================================================
# CREATE / LOAD INDEX
# =========================================================

index = create_index(
    nodes
)


# =========================================================
# NOVATECH RETRIEVER
# =========================================================

retriever = create_retriever(
    index,
    "novatech",
)


# =========================================================
# RERANKER
# =========================================================

reranker = create_reranker()


# =========================================================
# METRICS
# =========================================================

baseline_hit_scores = []
baseline_recall_scores = []
baseline_mrr_scores = []

reranked_hit_scores = []
reranked_recall_scores = []
reranked_mrr_scores = []

unanswerable_results = []
company_isolation_results = []


# =========================================================
# CATEGORY RESULTS
# =========================================================

category_results = {}


# =========================================================
# RUN TESTS
# =========================================================

for test in TEST_CASES:

    test_id = test["id"]

    category = test["category"]

    question = test["question"]

    expected_documents = test[
        "expected_documents"
    ]


    print(
        "\n" + "=" * 70
    )

    print(
        f"TEST {test_id} — "
        f"{category.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        "QUESTION:",
        question,
    )


    if expected_documents:

        print(
            "EXPECTED DOCUMENTS:",
            ", ".join(
                expected_documents
            ),
        )

    else:

        print(
            "EXPECTED DOCUMENTS: NONE"
        )


    # =====================================================
    # 1. BASELINE VECTOR RETRIEVAL
    # =====================================================

    vector_nodes = retriever.retrieve(
        question
    )


    # Remove duplicate chunks so
    # evaluation happens at document level.

    vector_nodes = remove_duplicate_chunks(
        vector_nodes
    )


    baseline_documents = [

        get_document_name(node)

        for node in vector_nodes
    ]


    # =====================================================
    # 2. RERANKING
    # =====================================================

    reranked_nodes = rerank_nodes(
        question,
        vector_nodes,
        reranker,
    )


    reranked_documents = [

        get_document_name(node)

        for node in reranked_nodes
    ]


    # =====================================================
    # 3. FINAL CONTEXT SELECTION
    # =====================================================

    final_nodes = select_relevant_nodes(
        question,
        reranked_nodes,
    )


    final_documents = [

        get_document_name(node)

        for node in final_nodes
    ]


    # =====================================================
    # PRINT BASELINE
    # =====================================================

    print_ranking(
        "BASELINE VECTOR RANKING:",
        vector_nodes,
    )


    # =====================================================
    # PRINT RERANKED
    # =====================================================

    print_ranking(
        "RERANKED RANKING:",
        reranked_nodes,
        show_reranker_score=True,
    )


    # =====================================================
    # PRINT FINAL CONTEXT
    # =====================================================

    print(
        "\nFINAL SELECTED DOCUMENTS:"
    )


    if final_documents:

        for document in final_documents:

            print(
                f"- {document}"
            )

    else:

        print(
            "- None"
        )


    # =====================================================
    # STANDARD METRICS
    # =====================================================

    baseline_hit = hit_at_1(
        baseline_documents,
        expected_documents,
    )

    baseline_recall = recall_at_3(
        baseline_documents,
        expected_documents,
    )

    baseline_rr = reciprocal_rank(
        baseline_documents,
        expected_documents,
    )


    reranked_hit = hit_at_1(
        reranked_documents,
        expected_documents,
    )

    reranked_recall = recall_at_3(
        reranked_documents,
        expected_documents,
    )

    reranked_rr = reciprocal_rank(
        reranked_documents,
        expected_documents,
    )


    # =====================================================
    # SPECIAL METRICS
    # =====================================================

    special_result = None


    if category == "unanswerable":

        special_result = (
            evaluate_unanswerable(
                final_documents
            )
        )

        unanswerable_results.append(
            special_result
        )


    elif category == "company_isolation":

        special_result = (
            evaluate_company_isolation(
                final_documents
            )
        )

        company_isolation_results.append(
            special_result
        )


    # =====================================================
    # STORE STANDARD METRICS
    # =====================================================

    if baseline_hit is not None:

        baseline_hit_scores.append(
            baseline_hit
        )

        baseline_recall_scores.append(
            baseline_recall
        )

        baseline_mrr_scores.append(
            baseline_rr
        )


    if reranked_hit is not None:

        reranked_hit_scores.append(
            reranked_hit
        )

        reranked_recall_scores.append(
            reranked_recall
        )

        reranked_mrr_scores.append(
            reranked_rr
        )


    # =====================================================
    # CATEGORY STORAGE
    # =====================================================

    if category not in category_results:

        category_results[
            category
        ] = {

            "baseline_hit": [],

            "reranked_hit": [],

            "baseline_mrr": [],

            "reranked_mrr": [],

            "special_results": [],
        }


    if baseline_hit is not None:

        category_results[
            category
        ]["baseline_hit"].append(
            baseline_hit
        )

        category_results[
            category
        ]["baseline_mrr"].append(
            baseline_rr
        )


    if reranked_hit is not None:

        category_results[
            category
        ]["reranked_hit"].append(
            reranked_hit
        )

        category_results[
            category
        ]["reranked_mrr"].append(
            reranked_rr
        )


    if special_result is not None:

        category_results[
            category
        ]["special_results"].append(
            special_result
        )


    # =====================================================
    # PRINT TEST METRICS
    # =====================================================

    print(
        "\nTEST METRICS:"
    )


    if baseline_hit is not None:

        print(
            f"Baseline Hit@1: "
            f"{baseline_hit}"
        )

        print(
            f"Baseline Recall@3: "
            f"{baseline_recall:.3f}"
        )

        print(
            f"Baseline RR: "
            f"{baseline_rr:.3f}"
        )

        print(
            f"Reranked Hit@1: "
            f"{reranked_hit}"
        )

        print(
            f"Reranked Recall@3: "
            f"{reranked_recall:.3f}"
        )

        print(
            f"Reranked RR: "
            f"{reranked_rr:.3f}"
        )


    elif category == "unanswerable":

        print(
            "Expected behavior: "
            "no supporting context."
        )

        if special_result == 1:

            print(
                "Unanswerable rejection: PASS"
            )

        else:

            print(
                "Unanswerable rejection: FAIL"
            )


    elif category == "company_isolation":

        print(
            "Expected behavior: "
            "no Google documents in "
            "NovaTech context."
        )

        if special_result == 1:

            print(
                "Company isolation: PASS"
            )

        else:

            print(
                "Company isolation: FAIL"
            )


# =========================================================
# FINAL EVALUATION
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL EVALUATION"
)

print(
    "=" * 70
)


# =========================================================
# BASELINE
# =========================================================

print(
    "\nBASELINE VECTOR RETRIEVAL"
)

print(
    f"Hit@1:       "
    f"{average(baseline_hit_scores) * 100:.2f}%"
)

print(
    f"Recall@3:    "
    f"{average(baseline_recall_scores) * 100:.2f}%"
)

print(
    f"MRR:         "
    f"{average(baseline_mrr_scores):.3f}"
)


# =========================================================
# RERANKED
# =========================================================

print(
    "\nRERANKED RETRIEVAL"
)

print(
    f"Hit@1:       "
    f"{average(reranked_hit_scores) * 100:.2f}%"
)

print(
    f"Recall@3:    "
    f"{average(reranked_recall_scores) * 100:.2f}%"
)

print(
    f"MRR:         "
    f"{average(reranked_mrr_scores):.3f}"
)


# =========================================================
# UNANSWERABLE
# =========================================================

print(
    "\nUNANSWERABLE EVALUATION"
)

print(
    f"Correctly rejected: "
    f"{average(unanswerable_results) * 100:.2f}%"
)


# =========================================================
# COMPANY ISOLATION
# =========================================================

print(
    "\nCOMPANY ISOLATION EVALUATION"
)

print(
    f"No Google document leakage: "
    f"{average(company_isolation_results) * 100:.2f}%"
)


# =========================================================
# CATEGORY RESULTS
# =========================================================

print(
    "\nCATEGORY RESULTS"
)


for category, results in (
    category_results.items()
):

    print(
        f"\n{category.upper()}"
    )


    if results["baseline_hit"]:

        print(
            f"Baseline Hit@1: "
            f"{average(results['baseline_hit']) * 100:.2f}%"
        )

        print(
            f"Reranked Hit@1: "
            f"{average(results['reranked_hit']) * 100:.2f}%"
        )

        print(
            f"Baseline MRR: "
            f"{average(results['baseline_mrr']):.3f}"
        )

        print(
            f"Reranked MRR: "
            f"{average(results['reranked_mrr']):.3f}"
        )


    elif results["special_results"]:

        if category == "unanswerable":

            print(
                f"Correctly rejected: "
                f"{average(results['special_results']) * 100:.2f}%"
            )


        elif category == "company_isolation":

            print(
                f"No Google leakage: "
                f"{average(results['special_results']) * 100:.2f}%"
            )


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "EVALUATION COMPLETE"
)

print(
    "=" * 70
)