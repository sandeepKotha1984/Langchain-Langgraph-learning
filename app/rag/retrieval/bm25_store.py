from rank_bm25 import BM25Okapi
import pickle

from app.config.config import BM25_STORE_PATH


class BM25Store:
    def __init__(self):
        self.store = None

    def create_bm25_store(self, chunks,document_content):
        retrieval_chunks = [
            chunk
            for chunk in chunks
            if not chunk.page_content.startswith("9. Evaluation Questions")
        ]

        tokenized_chunks = [
            chunk.page_content.lower().split()
            for chunk in retrieval_chunks
        ]

        print("Number of chunks:", len(retrieval_chunks))
        print("Tokenized chunks:", tokenized_chunks)

        bm25 = BM25Okapi(tokenized_chunks)

        bm25_store = {
            "bm25": bm25,
            "chunks": retrieval_chunks
        }

        with open(BM25_STORE_PATH, "wb") as f:
            pickle.dump(bm25_store, f)

        with open("./app/data/bm25_documents.pkl", "wb") as f:
            pickle.dump(document_content, f)


        self.store = bm25_store

        return bm25_store

    def get_store(self):
        if self.store:
            return self.store
        else:
            return None

