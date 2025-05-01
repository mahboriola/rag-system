from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra="ignore"
    )

    OPENAI_ENDPOINT: str
    OPENAI_API_KEY: str
    OPENAI_API_VERSION: str
    OPENAI_TYPE: str
    OPENAI_CHAT_MODEL: str
    OPENAI_EMBEDDING_MODEL: str

    QDRANT_ENDPOINT: str
    QDRANT_PORT: int
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str

    VECTOR_DIMENSIONS: int = 1536

    AZURE_OCR_ENDPOINT: str
    AZURE_OCR_KEY: str
