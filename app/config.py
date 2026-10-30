import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "NexusDocs Advanced RAG"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis-cache:6379/0")
    
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    TOP_K_RETRIEVAL: int = 6
    TOP_K_RERANK: int = 3

    class Config:
        env_file = ".env"

settings = Settings()