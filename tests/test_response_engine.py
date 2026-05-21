"""Tests for the ResponseEngine (RAG mode)."""

import pytest

from hbr_apple_rag.response_engine import ResponseEngine


def test_response_engine_rag_returns_valid_answer():
    """Basic smoke test for RAG mode."""
    engine = ResponseEngine(mode="rag")
    answer = engine.respond("Who are the authors of the article?")

    assert isinstance(answer, str)
    assert len(answer) > 30
    # Should contain at least one of the expected key pieces from the article
    assert any(
        keyword in answer
        for keyword in ["Podolny", "Hansen", "Harvard Business Review"]
    )


def test_response_engine_rag_handles_different_questions():
    """Test that RAG mode can handle different questions without crashing."""
    engine = ResponseEngine(mode="rag")
    answer = engine.respond("What is the main idea of the article?")

    assert isinstance(answer, str)
    assert len(answer) > 20


def test_response_engine_rag_mode_initialization():
    """Test that ResponseEngine can be initialized in RAG mode."""
    engine = ResponseEngine(mode="rag")
    assert engine.mode == "rag"
    assert engine._retriever is not None