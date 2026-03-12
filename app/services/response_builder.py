from typing import List

def format_response(answer: str, chunks: List[dict], similarity_scores: List[float]) -> dict:

    sources = []

    for chunk in chunks:
        sources.append(
            {
                "source": chunk.get("source"),
                "page": chunk.get("page")
            }
        )

    confidence = 0.0

    if similarity_scores:
        confidence = round(sum(similarity_scores) / len(similarity_scores), 4)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }