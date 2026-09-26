from langchain_community.retrievers import BM25Retriever
from langsmith import traceable


@traceable(name="bm25_retrieval")
def create_bm25_retriever(chunks, top_k=5):
    retriever = BM25Retriever.from_documents(
        chunks,
        k=top_k
    )

    return retriever


@traceable(name="bm25_search")
def search_bm25(retriever, query):
    return retriever.invoke(query)