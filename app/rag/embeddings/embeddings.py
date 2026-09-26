from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings(chunks):
    """
    Load embeddings for the given chunks using SentenceTransformer.

    Args:
        chunks (list): A list of text chunks for which to generate embeddings.

    Returns:
        list: A list of embedding vectors.
    """
    embeddings = model.encode([chunk.page_content for chunk in chunks])
    return embeddings,model

def query_embeddings(query):
    """
    Generate embeddings for the given query using SentenceTransformer.

    Args:
        query (str): The search query.
    """
    query_embedding = model.encode([query])
    return query_embedding