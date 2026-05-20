"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === OpenAI Configuration ===
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        env="OPENAI_API_BASE"
    )

    # === Model Configuration ===
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )

    # === RAG Pipeline Parameters (aligned with notebook) ===
    chunk_size: int = Field(default=256, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=25, env="CHUNK_OVERLAP")
    retriever_k: int = Field(default=3, env="RETRIEVER_K")

    # === Generation Parameters (from notebook) ===
    temperature: float = Field(default=0.3, env="TEMPERATURE")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    top_p: float = Field(default=0.95, env="TOP_P")

    # === Paths ===
    data_dir: Path = Field(default=Path("data/raw"), env="DATA_DIR")
    vector_store_dir: Path = Field(
        default=Path("articles_db"),
        env="VECTOR_STORE_DIR"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()