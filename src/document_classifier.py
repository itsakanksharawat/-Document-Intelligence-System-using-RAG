from pathlib import Path


DOCUMENT_TYPES = {
    "leave_policy": [
        "leave",
        "vacation",
        "casual leave",
        "sick leave",
        "earned leave",
        "annual leave",
    ],
    "expense_policy": [
        "expense",
        "reimbursement",
        "travel expense",
        "claim",
        "business expense",
    ],
    "it_security_policy": [
        "security",
        "password",
        "cybersecurity",
        "information security",
        "data protection",
    ],
    "code_of_conduct": [
        "code of conduct",
        "ethics",
        "professional conduct",
        "employee conduct",
    ],
    "onboarding_guide": [
        "onboarding",
        "new employee",
        "joining",
        "orientation",
    ],
    "remote_work_policy": [
        "remote work",
        "work from home",
        "wfh",
        "remote working",
    ],
    "hybrid_work_handbook": [
        "hybrid work",
        "hybrid working",
        "workplace flexibility",
    ],
}


def classify_document(
    file_path,
    text,
):
    """
    Classify a document using its actual content.

    Filename is used only as a fallback.
    """

    if not text:
        text = ""

    text_lower = text.lower()

    # ---------------------------------------------
    # CLASSIFY USING DOCUMENT CONTENT
    # ---------------------------------------------

    scores = {}

    for document_type, keywords in DOCUMENT_TYPES.items():

        score = 0

        for keyword in keywords:

            if keyword in text_lower:
                score += 1

        scores[document_type] = score

    best_type = max(
        scores,
        key=scores.get,
    )

    best_score = scores[best_type]

    if best_score > 0:
        return best_type

    # ---------------------------------------------
    # FILENAME FALLBACK
    # ---------------------------------------------

    filename = Path(file_path).name.lower()

    for document_type, keywords in DOCUMENT_TYPES.items():

        for keyword in keywords:

            if keyword in filename:
                return document_type

    return "unknown"