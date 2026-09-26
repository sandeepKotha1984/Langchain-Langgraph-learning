from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct,FieldCondition,Filter,MatchValue
from langsmith import traceable
from pathlib import Path
from qdrant_client import QdrantClient

from app.config.config import QDRANT_PATH

client = QdrantClient(
    path=str(QDRANT_PATH)
)

COLLECTION_NAME = "enterprise_rag"



def create_collection(vector_size=384):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def insert_embeddings(chunks, embeddings):
    points = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(chunk.page_content)
        points.append(
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.page_content,
                    "metadata": {"chunk_id": idx}
                }
            )
        )
    print(f"Inserting {len(points)} points into Qdrant collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

@traceable(name="vector_search")
def search(query_embedding, top_k=3):
    query_vector = query_embedding.flatten().tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    #print(results)
    return results.points


def get_collections():
    """
    Return available Qdrant collections.
    Useful for debugging.
    """
    return client.get_collections()


def close_client():
    """
    Close the Qdrant client and release the local storage lock.
    """
    client.close()