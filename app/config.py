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
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.7
    
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
