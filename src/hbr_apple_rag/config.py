"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === OpenAI Configuration ===
    openai_api_key: str
    openai_api_base: str

    # === Model Configuration ===
    llm_model: str
    embedding_model: str

    # === RAG Pipeline Parameters (aligned with notebook) ===
    chunk_size: int
    chunk_overlap: int
    retriever_k: int

    # === Generation Parameters (from notebook) ===
    temperature: float
    max_tokens: int
    top_p: float

    # === Paths ===
    data_dir: Path
    vector_store_dir: Path

    # === Retriever MMR Parameters ===
    retriever_fetch_k: int
    retriever_lambda_mult: float


# Global settings instance
settings = Settings()