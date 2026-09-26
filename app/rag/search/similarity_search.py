from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.ext.asyncio import result


def search(query, model, embeddings, chunks, top_k=5):
    """
    Perform a similarity search on the given index using the provided query.

    Args:
        query (str): The search query.
        model (SentenceTransformer): The sentence transformer model.
        embeddings (list): The list of embedding vectors.
        top_k (int): The number of top results to return.

    Returns:
        list: A list of the top_k most similar items from the index.
    """
    # Preprocess the query
    encoded_query = model.encode(query)

    # Compute similarity scores
    scores = cosine_similarity([encoded_query], embeddings)[0]

    print("Scores:", scores)


    indices = np.argsort(scores)[::-1][:top_k]  # Get the indices of the top_k scores

    # Get the top_k results
    results = [
        (chunks[i], scores[i])
        for i in indices
    ]

    return results