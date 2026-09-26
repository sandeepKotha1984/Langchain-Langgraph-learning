

from rag.embeddings.embeddings import load_embeddings
from rag.chunking.chunking import chunk_documents
from rag.loaders.document_loader import load_document
from rag.retrieval.bm25_retreival import create_bm25_retriever,search_bm25
from rag.retrieval.bm25_store import BM25Store
from rag.search.similarity_search import search
from rag.vector_store.qDrant import insert_embeddings, create_collection


def main():
    # Example usage of the load_document function
    file_path = "enterprise_rag_mixed_structure_policy.docx"  # Replace with your actual file path
    document_content = load_document(file_path)
    bm_25_store = BM25Store()

    #print(document_content)
    chunked_documents = chunk_documents(document_content, chunk_size=300, chunk_overlap=30)
    print(f"Number of chunks created: {type(chunked_documents)}")

    #for i, chunk in enumerate(chunked_documents):
        #print(f"Chunk {i + 1}: {chunk}")

    embedding_list, model = load_embeddings(chunked_documents)
    print(f"Embeddings shape: {embedding_list.shape}")
    query = "What approvals are required for an INR 75,000 expense?"

    results = search(query, model, embedding_list, chunked_documents, top_k=5)
    print(f"Results: {results}")
    create_collection()
    insert_embeddings(chunked_documents, embedding_list)
    bm25_store = bm_25_store.create_bm25_store(chunked_documents,document_content)

    print(f"BM25 store created with {bm25_store} documents.")


    index_search = search_bm25(bm25_store, "What approvals are required for an INR 75,000 expense?", top_k=5)

    #print(f"BM25 search results: {index_search}")

if __name__ == "__main__":
    main()