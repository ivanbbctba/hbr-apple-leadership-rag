"""Tests for configuration management."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from hbr_apple_rag.config import Settings


@pytest.fixture
def mock_env(monkeypatch):
    """Set mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "dummy_api_key",
        "OPENAI_API_BASE": "https://mock-api.openai.com/v1",
        "LLM_MODEL": "mock-model",
        "EMBEDDING_MODEL": "mock-embedding",
        "CHUNK_SIZE": "512",
        "CHUNK_OVERLAP": "64",
        "RETRIEVER_K": "10",
        "TEMPERATURE": "0.7",
        "MAX_TOKENS": "2048",
        "TOP_P": "0.9",
        "DATA_DIR": "mock_data/raw",
        "VECTOR_STORE_DIR": "mock_db",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


def test_settings_load_from_env(mock_env):
    """Test that settings correctly load and parse environment variables."""
    settings = Settings()

    assert settings.openai_api_key == "dummy_api_key"
    assert settings.openai_api_base == "https://mock-api.openai.com/v1"
    assert settings.llm_model == "mock-model"
    assert settings.embedding_model == "mock-embedding"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.retriever_k == 10
    assert settings.temperature == 0.7
    assert settings.max_tokens == 2048
    assert settings.top_p == 0.9
    assert settings.data_dir == Path("mock_data/raw")
    assert settings.vector_store_dir == Path("mock_db")


def test_settings_fails_without_required_vars(monkeypatch):
    """Test that Settings fails when required environment variables are missing."""
    # Ensure critical variables are not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with pytest.raises(ValidationError):
        _ = Settings(_env_file=None)


def test_settings_type_enforcement(mock_env):
    """Verify that settings enforce correct Python types."""
    settings = Settings()

    assert isinstance(settings.chunk_size, int)
    assert isinstance(settings.chunk_overlap, int)
    assert isinstance(settings.retriever_k, int)
    assert isinstance(settings.temperature, float)
    assert isinstance(settings.max_tokens, int)
    assert isinstance(settings.top_p, float)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.vector_store_dir, Path)