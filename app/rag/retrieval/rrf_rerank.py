from collections import defaultdict


def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    top_k=10,
    k=60
):
    scores = defaultdict(float)
    documents = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result.payload["metadata"]["chunk_id"]

        scores[chunk_id] += 1 / (k + rank)
        documents[chunk_id] = result.payload

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.metadata["chunk_id"]

        scores[chunk_id] += 1 / (k + rank)

        # Keep document if vector search didn't provide it
        documents.setdefault(
            chunk_id,
            result
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "chunk_id": chunk_id,
            "rrf_score": score,
            **normalize_document(documents[chunk_id])
        }
        for chunk_id, score in ranked[:top_k]
    ]

def normalize_document(document):
    if isinstance(document, dict):
        return {
            "text": document["text"],
            "metadata": document["metadata"]
        }

    return {
        "text": document.page_content,
        "metadata": document.metadata
    }