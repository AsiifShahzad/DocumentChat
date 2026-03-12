from typing import List


def build_prompt(chunks: List[dict], question: str) -> str:

    context_parts = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        context_parts.append(f"[Chunk {i+1} | Source: {source}, Page: {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say "I couldn't find relevant information in the document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt
