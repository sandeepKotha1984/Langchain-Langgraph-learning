import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "app" / "data"

BM25_STORE_PATH = DATA_DIR / "bm25_store.pkl"
QDRANT_PATH = DATA_DIR / "qdrant_data"

load_dotenv()

JINA_RERANKER_API_KEY = os.getenv("JINA_RERANKER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")