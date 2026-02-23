"""
Application configuration settings
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # 'openai' or 'ollama'
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama2"  # Default model for Ollama
    LLM_TEMPERATURE: float = 0.7
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"  # Can be any Ollama model: mistral, neural-chat, etc
    
    # Vector Store Settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Data Paths
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
