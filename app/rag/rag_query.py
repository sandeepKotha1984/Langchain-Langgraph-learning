from app.rag.config.config import BM25_STORE_PATH
from app.rag.embeddings.embeddings import query_embeddings
from app.rag.llm.genai_client import LLMClient
from app.rag.prompt_builder.prompt import build_prompt
from app.rag.reranker_results.reranker_results import rerank_results
from app.rag.retrieval.bm25_store import BM25Store
from app.rag.vector_store.qDrant import search
from app.rag.retrieval.bm25_retreival import (
    create_bm25_retriever,
    search_bm25
)
from langsmith import traceable
from app.rag.retrieval.rrf_rerank import reciprocal_rank_fusion
import pickle

@traceable(name="enterprise-rag")
def rag_query(query=None):
    if query is None:
        query = input("Enter query: ")

    bm_25_store = BM25Store()
    llm = LLMClient()  # Placeholder for LLM client initialization

    # Query embedding
    query_embedding = query_embeddings(query)

    print(f"\nQuery embedding shape: {query_embedding}")


    # Vector search using query_embedding

    vector_search_results = search(query_embedding)  # Placeholder for vector search results
    print(f"\nvector Query results: {vector_search_results}")


    # BM25 search
    #Read from pickle file and create BM25 retriever

    with open(BM25_STORE_PATH, "rb") as f:
        bm_25_stored = pickle.load(f)



    retriever = create_bm25_retriever(bm_25_stored["chunks"])
    print(f"BM25 store loaded with {bm_25_stored} documents.")

    bm25_search = search_bm25(retriever,query)

    print(f"BM25 search results: {bm25_search}")






    # Jina reranking
    results = rerank_results(vector_search_results, bm25_search,query)
    print(f"re rank  results: {results}")

    #PROMPT_BUILDING
    prompt = build_prompt(query, results)
    print(f"Generated prompt: {prompt}")


    # LLM generation
    llm_response = llm.generate_answer(prompt)
    print(f"LLM response: {llm_response}")

    return {
        "answer": llm_response,
        "vector_search_results": vector_search_results,
        "bm25_search_results": bm25_search,
        "reranked_results": results
    }


