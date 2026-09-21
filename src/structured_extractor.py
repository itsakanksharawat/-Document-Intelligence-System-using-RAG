import json

from src.generator import create_llm


def build_extraction_prompt(
    document_text,
    document_type,
):
    """
    Build a prompt for extracting structured
    information from a document.
    """

    prompt = f"""
You are a document intelligence system.

Extract structured information from the
document below.

DOCUMENT TYPE:
{document_type}

IMPORTANT RULES:

1. Use ONLY information explicitly present
   in the document.
2. Do not invent or infer missing values.
3. If a field is not present, use null.
4. Preserve numbers, dates, names, and limits
   exactly as stated when possible.
5. Return ONLY valid JSON.
6. Do not include markdown.
7. Do not include explanations outside JSON.

DOCUMENT:
{document_text}

Return JSON using this structure:

{{
    "document_type": "{document_type}",
    "key_information": {{}}
}}
"""

    return prompt


def extract_structured_data(
    document_text,
    document_type,
    llm=None,
):
    """
    Extract structured information from
    document text using Gemini.
    """

    if not document_text:
        return {
            "document_type": document_type,
            "key_information": {},
        }

    if llm is None:
        llm = create_llm()

    prompt = build_extraction_prompt(
        document_text=document_text,
        document_type=document_type,
    )

    response = llm.complete(prompt)

    response_text = str(response).strip()

    # Remove accidental markdown code fences
    if response_text.startswith("```"):
        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:
        extracted_data = json.loads(
            response_text
        )
    except json.JSONDecodeError:
        return {
            "document_type": document_type,
            "key_information": {},
            "raw_response": response_text,
            "error": "LLM did not return valid JSON.",
        }

    return extracted_data