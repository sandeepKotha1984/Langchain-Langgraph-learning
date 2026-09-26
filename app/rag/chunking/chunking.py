from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_documents(documents, chunk_size, chunk_overlap):
    """
    Chunk the documents into smaller pieces of specified size.

    Args:
        documents (list): List of documents to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks=text_splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata = {"chunk_id": idx}
    return chunks