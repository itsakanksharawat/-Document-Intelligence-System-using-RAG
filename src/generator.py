from llama_index.llms.google_genai import GoogleGenAI

from src.config import GOOGLE_API_KEY, LLM_MODEL


def create_llm():
    llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
    )

    print("Gemini LLM loaded.")

    return llm


def build_prompt(question, nodes):
    evidence_blocks = []

    for index, node in enumerate(nodes, start=1):

        source_file = node.metadata.get(
            "source_file",
            "Unknown source",
        )

        page_number = node.metadata.get(
            "page_number"
        )

        if page_number:
            source = (
                f"{source_file}, "
                f"page {page_number}"
            )
        else:
            source = source_file

        text = node.get_content()

        evidence_blocks.append(
            f"""
EVIDENCE {index}
SOURCE: {source}

{text}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    prompt = f"""
You are a company knowledge assistant.

Answer the user's question using ONLY
the evidence provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not assume information that is not
   explicitly supported by the evidence.
4. If the evidence does not contain enough
   information to answer the question,
   say:

   "I don't have enough information in
   the provided documents to answer this."

5. Keep the answer concise and factual.
6. Answer only the user's question.
7. Do not include a Sources section.
8. Do not cite or mention document names
   in the answer.

USER QUESTION:
{question}

EVIDENCE:
{evidence}

FORMAT:

Answer:
<answer>
"""

    return prompt

def generate_answer(
    question,
    nodes,
    llm=None,
):
    if not nodes:
        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents to "
                "answer this."
            ),
            "sources": [],
        }

    if llm is None:
        llm = create_llm()

    prompt = build_prompt(
        question,
        nodes,
    )

    response = llm.complete(prompt)

    answer_text = str(response)

    sources = []

    for node in nodes:

        source_file = node.metadata.get(
            "source_file",
            "Unknown source",
        )

        page_number = node.metadata.get(
            "page_number"
        )

        if page_number:
            source = (
                f"{source_file}, "
                f"page {page_number}"
            )
        else:
            source = source_file

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer_text,
        "sources": sources,
    }