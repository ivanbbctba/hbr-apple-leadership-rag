"""Tests for evaluation module (LLM-as-Judge)."""

import pytest
from unittest.mock import patch, MagicMock

from src.hbr_apple_rag.evaluation import evaluate_faithfulness, evaluate_relevance


class TestEvaluation:
    @patch("src.hbr_apple_rag.evaluation.ChatOpenAI")
    def test_evaluate_faithfulness_returns_float(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "4"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        score = evaluate_faithfulness(
            question="Who are the authors?",
            answer="The authors are Podolny and Hansen.",
            context="The article was written by Joel M. Podolny and Morten T. Hansen."
        )

        assert isinstance(score, float)
        assert 0 <= score <= 5

    @patch("src.hbr_apple_rag.evaluation.ChatOpenAI")
    def test_evaluate_relevance_returns_float(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "5"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        score = evaluate_relevance(
            question="Who are the authors?",
            answer="Joel M. Podolny and Morten T. Hansen wrote the article."
        )

        assert isinstance(score, float)
        assert 0 <= score <= 5

    @patch("src.hbr_apple_rag.evaluation.ChatOpenAI")
    def test_evaluate_faithfulness_handles_bad_response(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "not a number"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        score = evaluate_faithfulness("Q", "A", "C")
        assert score == 0.0

    @patch("src.hbr_apple_rag.evaluation.ChatOpenAI")
    def test_evaluate_faithfulness_with_empty_context(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "2"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        score = evaluate_faithfulness("Q", "A", "")
        assert isinstance(score, float)

    @patch("src.hbr_apple_rag.evaluation.ChatOpenAI")
    def test_evaluate_relevance_with_very_long_answer(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "3"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        long_answer = "x" * 2000
        score = evaluate_relevance("Short question", long_answer)
        assert 0 <= score <= 5