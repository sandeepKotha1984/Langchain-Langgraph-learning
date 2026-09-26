from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.rankers.jina import JinaRanker
from langsmith import traceable

from app.config.config import JINA_RERANKER_API_KEY
from app.retrieval.rrf_rerank import reciprocal_rank_fusion


ranker = JinaRanker(
    api_key=Secret.from_token(JINA_RERANKER_API_KEY)
)

@traceable(name="rerank_results")
def rerank_results(vector_search_results, bm25_search, query):
    """
    Combine vector and BM25 results using RRF,
    then rerank the candidates using Jina.
    """

    # 1. Hybrid retrieval
    rrf_results = reciprocal_rank_fusion(
        vector_search_results,
        bm25_search,
        top_k=5
    )

    # 2. Convert RRF results to Haystack Documents
    documents = [
        Document(
            content=result["text"],
            meta={
                **result["metadata"],
                "chunk_id": result["chunk_id"],
                "rrf_score": result["rrf_score"]
            }
        )
        for result in rrf_results
    ]

    # 3. Jina reranking
    response = ranker.run(
        query=query,
        documents=documents
    )

    return response["documents"]