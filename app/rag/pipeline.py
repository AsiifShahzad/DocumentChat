from app.rag.retriever import query_embedding, retrieve_chunks
from app.services.re_ranker import rerank_chunks


def retrieve_context(query: str):

    # Step 1: Embed the query
    query_vector = query_embedding(query)

    # Step 2: Vector search — wide net, top 20 candidates
    candidates, _ = retrieve_chunks(query_vector, top_k=20)

    # Step 3: Rerank — narrow to top 5 accurately
    final_chunks = rerank_chunks(query, candidates, top_k=5)

    return final_chunks