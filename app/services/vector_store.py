from typing import List
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv() 

INDEX_NAME = "bge-small-index"
DIMENSION = 384

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def insert_embeddings(embeddings: List[List[float]], metadata: List[dict]):
    vectors = []
    for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
        vectors.append(
            {
                "id": str(i),
                "values": embedding,
                "metadata": meta
            }
        )
    index.upsert(vectors=vectors)

def similarity_search(query_embedding: List[float], top_k: int = 20) -> dict:
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results