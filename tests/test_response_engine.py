"""Tests for the ResponseEngine class.

These tests validate that the ResponseEngine works correctly across
its three modes (raw, prompt_eng, and rag). The goal is to ensure
that the class behaves as expected and that the three approaches
can be compared fairly.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.hbr_apple_rag.response_engine import ResponseEngine


class TestResponseEngine:
    def test_invalid_mode_raises_error(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ResponseEngine(mode="invalid")

    def test_mode_is_set_correctly(self):
        engine = ResponseEngine(mode="raw")
        assert engine.mode == "raw"

    @patch("src.hbr_apple_rag.response_engine.ChatOpenAI")
    def test_raw_mode_returns_string(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "This is a test answer from the model."
        mock_chat_openai.return_value.invoke.return_value = mock_response

        engine = ResponseEngine(mode="raw")
        result = engine.respond("Test question")

        assert isinstance(result, str)
        assert len(result) > 5

    @patch("src.hbr_apple_rag.response_engine.ChatOpenAI")
    def test_prompt_engineering_mode_uses_system_prompt(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "Answer with system prompt."
        mock_chat_openai.return_value.invoke.return_value = mock_response

        engine = ResponseEngine(mode="prompt_eng")
        result = engine.respond("Test question")

        # Check that invoke was called with messages (system + human)
        call_args = mock_chat_openai.return_value.invoke.call_args[0][0]
        assert any(getattr(msg, "type", None) == "system" for msg in call_args)

    def test_rag_mode_initializes_retriever(self):
        engine = ResponseEngine(mode="rag")
        assert engine.mode == "rag"
        assert engine._retriever is not None

    @patch("src.hbr_apple_rag.response_engine.ChatOpenAI")
    def test_respond_captures_usage(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "Answer with usage data."
        mock_response.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 30,
            "total_tokens": 80,
        }
        mock_chat_openai.return_value.invoke.return_value = mock_response

        engine = ResponseEngine(mode="raw")
        engine.respond("Hello")

        usage = engine.get_last_usage()
        assert usage.get("total_tokens") == 80

    @patch("src.hbr_apple_rag.response_engine.ChatOpenAI")
    def test_get_last_usage_returns_data_after_respond(self, mock_chat_openai):
        mock_response = MagicMock()
        mock_response.content = "Answer text"
        mock_response.usage_metadata = {
            "input_tokens": 45,
            "output_tokens": 25,
            "total_tokens": 70,
        }
        mock_chat_openai.return_value.invoke.return_value = mock_response

        engine = ResponseEngine(mode="raw")
        engine.respond("Hello")

        usage = engine.get_last_usage()
        assert usage["total_tokens"] == 70
        assert usage["input_tokens"] == 45

    @patch("src.hbr_apple_rag.response_engine.ChatOpenAI")
    def test_respond_handles_llm_exception(self, mock_chat_openai):
        mock_chat_openai.return_value.invoke.side_effect = Exception("API rate limit")

        engine = ResponseEngine(mode="raw")

        with pytest.raises(Exception):
            engine.respond("This should fail")