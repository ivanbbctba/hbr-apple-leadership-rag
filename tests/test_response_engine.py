"""Tests for the ResponseEngine class.

These tests validate that the ResponseEngine works correctly across
its three modes (raw, prompt_eng, and rag). The goal is to ensure
that the class behaves as expected and that the three approaches
can be compared fairly.
"""

import pytest

from hbr_apple_rag.response_engine import ResponseEngine


class TestResponseEngine:
    """Test suite for ResponseEngine class."""

    def test_raw_mode_returns_string(self):
        """Test that raw mode returns a non-empty string."""
        engine = ResponseEngine(mode="raw")
        answer = engine.respond("Who are the authors of the article?")

        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_prompt_eng_mode_returns_string(self):
        """Test that prompt engineering mode returns a non-empty string."""
        engine = ResponseEngine(mode="prompt_eng")
        answer = engine.respond("Who are the authors of the article?")

        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_rag_mode_returns_string(self):
        """Test that RAG mode returns a non-empty string."""
        engine = ResponseEngine(mode="rag")
        answer = engine.respond("Who are the authors of the article?")

        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_rag_mode_contains_relevant_information(self):
        """
        Basic check that RAG mode can retrieve useful information.

        This test verifies that the RAG mode is able to find relevant context
        and produce an answer that contains expected information from the article.
        """
        engine = ResponseEngine(mode="rag")
        answer = engine.respond("Who are the authors of the article?")

        # The answer should mention at least one of the key elements
        assert any(
            keyword.lower() in answer.lower()
            for keyword in ["podolny", "hansen", "harvard business review"]
        )

    def test_invalid_mode_raises_error(self):
        """Test that using an invalid mode raises a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ResponseEngine(mode="invalid_mode")

    def test_rag_mode_initializes_retriever(self):
        """Test that RAG mode properly initializes the retriever."""
        engine = ResponseEngine(mode="rag")
        assert engine.mode == "rag"
        assert engine._retriever is not None

    def test_different_modes_can_be_used_independently(self):
        """
        Test that we can create multiple ResponseEngine instances
        with different modes without interference.
        """
        raw_engine = ResponseEngine(mode="raw")
        rag_engine = ResponseEngine(mode="rag")

        raw_answer = raw_engine.respond("What is this article about?")
        rag_answer = rag_engine.respond("What is this article about?")

        assert isinstance(raw_answer, str)
        assert isinstance(rag_answer, str)