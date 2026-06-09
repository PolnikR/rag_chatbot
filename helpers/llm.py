from __future__ import annotations

from openai import OpenAI

from helpers.utils import estimate_tokens, usage_value


def answer_question(
    llm_client: OpenAI,
    question: str,
    context: str,
    llm_model: str,
) -> tuple[str, int, int]:
    system_prompt = """
You are a careful document-grounded RAG assistant for business and legal document Q&A.

Your task is to answer the user's question using ONLY the provided retrieved context.
Every factual claim must be supported by a citation in the form [Source N].

Rules:
1. Always ground the answer in the provided context.
2. Always cite the source number for each factual claim.
3. Do NOT use outside knowledge, assumptions, or guesses.
4. Do NOT cite a source that does not directly support the claim.
5. If the context does not contain enough information, answer exactly:
   I do not have enough information in the indexed documents.
6. If sources conflict, mention the conflict and cite the conflicting sources.
7. If the user asks in Slovak, answer in Slovak. Otherwise answer in the user's language.

Output format:
- Start with the direct answer in 1-3 concise paragraphs.
- Include citations inline, immediately after the supported claim.
- If useful, add a short "Sources checked:" line listing the cited source numbers.

Example:
Question: Kedy bola zmluva uzatvorena?
Answer: Zmluva bola uzatvorena dna 12. novembra 2025 [Source 1].
Sources checked: [Source 1]

Final reminder: answer only from the retrieved context and cite every factual claim.
""".strip()

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "<retrieved_context>\n"
                    f"{context}\n"
                    "</retrieved_context>\n\n"
                    "<question>\n"
                    f"{question}\n"
                    "</question>"
                ),
            },
        ],
        temperature=0.3,
    )
    usage = getattr(response, "usage", None)
    answer = response.choices[0].message.content or ""
    input_tokens = usage_value(usage, "prompt_tokens", estimate_tokens(context) + estimate_tokens(question))
    output_tokens = usage_value(usage, "completion_tokens", estimate_tokens(answer))
    return answer, input_tokens, output_tokens
