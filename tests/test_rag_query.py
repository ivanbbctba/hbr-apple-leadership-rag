"""Tests for the RAG Query Engine."""

import pytest

from hbr_apple_rag.rag_query import generate_rag_response


def test_generate_rag_response_returns_valid_answer():
    """Basic smoke test: function should return a non-empty string."""
    answer = generate_rag_response("Who are the authors of the article?")

    assert isinstance(answer, str)
    assert len(answer) > 30
    # The answer should contain at least one of the expected key pieces
    assert any(keyword in answer for keyword in ["Podolny", "Hansen", "Harvard Business Review"])


def test_generate_rag_response_handles_different_questions():
    """Test that the function can handle another question without crashing."""
    answer = generate_rag_response("What is the main idea of the article?")

    assert isinstance(answer, str)
    assert len(answer) > 20