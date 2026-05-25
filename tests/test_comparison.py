"""Tests for ComparisonEngine and ModeResult."""

import pytest
from unittest.mock import patch, MagicMock

from src.hbr_apple_rag.comparison import ComparisonEngine, ModeResult


class TestComparisonEngine:
    @patch("src.hbr_apple_rag.comparison.ResponseEngine")
    def test_run_mode_returns_mode_result(self, mock_response_engine):
        mock_engine = MagicMock()
        mock_engine.respond.return_value = "Test answer"
        mock_engine.get_last_context.return_value = []
        mock_engine.get_last_usage.return_value = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_response_engine.return_value = mock_engine

        engine = ComparisonEngine()
        result = engine.run_mode("raw", "Test question")

        assert isinstance(result, ModeResult)
        assert result.answer == "Test answer"
        assert result.total_tokens == 150
        assert result.prompt_tokens == 100

    def test_mode_result_has_evaluation_fields(self):
        result = ModeResult(
            answer="test",
            response_time=1.2,
            faithfulness_score=4.0,
            relevance_score=3.5,
        )
        assert result.faithfulness_score == 4.0
        assert result.relevance_score == 3.5

    @patch("src.hbr_apple_rag.comparison.ResponseEngine")
    def test_run_mode_includes_evaluation_scores(self, mock_response_engine):
        mock_engine = MagicMock()
        mock_engine.respond.return_value = "Some answer"

        # Simulate that we have context (important for RAG faithfulness)
        mock_doc = MagicMock()
        mock_doc.page_content = "The article was written by Podolny and Hansen."
        mock_engine.get_last_context.return_value = [mock_doc]

        mock_engine.get_last_usage.return_value = {"total_tokens": 200}
        mock_response_engine.return_value = mock_engine

        with (
            patch(
                "src.hbr_apple_rag.comparison.evaluate_faithfulness", return_value=4.5
            ),
            patch("src.hbr_apple_rag.comparison.evaluate_relevance", return_value=4.0),
        ):
            engine = ComparisonEngine()
            result = engine.run_mode("rag", "Test question")

            assert result.faithfulness_score == 4.5
            assert result.relevance_score == 4.0

    def test_mode_result_stores_tokens_correctly(self):
        result = ModeResult(
            answer="test",
            response_time=1.5,
            prompt_tokens=120,
            completion_tokens=80,
            total_tokens=200,
            estimated_cost_usd=0.00045,
        )
        assert result.total_tokens == 200
        assert result.estimated_cost_usd == 0.00045

    @patch("src.hbr_apple_rag.comparison.ResponseEngine")
    def test_raw_and_prompt_eng_modes_skip_faithfulness(self, mock_response_engine):
        mock_engine = MagicMock()
        mock_engine.respond.return_value = "Answer"
        mock_engine.get_last_context.return_value = []
        mock_engine.get_last_usage.return_value = {"total_tokens": 100}
        mock_response_engine.return_value = mock_engine

        engine = ComparisonEngine()
        result = engine.run_mode("raw", "Question")

        # Faithfulness should be None for non-RAG modes
        assert result.faithfulness_score is None
        assert result.relevance_score is not None

    @patch("src.hbr_apple_rag.comparison.ResponseEngine")
    def test_cost_is_calculated_when_tokens_present(self, mock_response_engine):
        mock_engine = MagicMock()
        mock_engine.respond.return_value = "Answer"
        mock_engine.get_last_context.return_value = []
        mock_engine.get_last_usage.return_value = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }
        mock_response_engine.return_value = mock_engine

        engine = ComparisonEngine()
        result = engine.run_mode("prompt_eng", "Q")

        assert result.estimated_cost_usd is not None
        assert result.estimated_cost_usd > 0

    @patch("src.hbr_apple_rag.comparison.ResponseEngine")
    @patch("src.hbr_apple_rag.comparison.evaluate_faithfulness")
    @patch("src.hbr_apple_rag.comparison.evaluate_relevance")
    def test_full_comparison_flow_end_to_end(
        self, mock_relevance, mock_faithfulness, mock_response_engine
    ):
        # Setup mocks
        mock_engine_instance = MagicMock()
        mock_engine_instance.respond.return_value = (
            "Apple uses a functional organizational structure led by experts."
        )
        mock_engine_instance.get_last_context.return_value = [
            MagicMock(page_content="The article discusses Apple's organization...")
        ]
        mock_engine_instance.get_last_usage.return_value = {
            "input_tokens": 180,
            "output_tokens": 95,
            "total_tokens": 275,
        }
        mock_response_engine.return_value = mock_engine_instance

        mock_faithfulness.return_value = 4.2
        mock_relevance.return_value = 4.8

        # Execute
        engine = ComparisonEngine()
        result = engine.run_mode("rag", "How is Apple organized?")

        # Assertions
        assert result.answer.startswith("Apple uses")
        assert result.total_tokens == 275
        assert result.estimated_cost_usd is not None
        assert result.estimated_cost_usd > 0
        assert result.faithfulness_score == 4.2
        assert result.relevance_score == 4.8
        assert len(result.context) == 1

class TestCostCalculation:
    def test_cost_calculation_with_known_model(self):
        engine = ComparisonEngine()
        cost = engine._calculate_cost(prompt_tokens=1000, completion_tokens=500)
        assert cost is not None
        assert cost > 0

    def test_cost_is_none_when_no_tokens(self):
        engine = ComparisonEngine()
        cost = engine._calculate_cost(prompt_tokens=None, completion_tokens=None)
        assert cost is None

    @pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-5.4-nano", "gpt-5.4-mini"])
    def test_cost_calculation_works_for_all_supported_models(self, model_name, monkeypatch):
        from src.hbr_apple_rag import config as config_module

        # Temporarily change the model
        original_model = config_module.settings.llm_model
        config_module.settings.llm_model = model_name

        engine = ComparisonEngine()
        cost = engine._calculate_cost(prompt_tokens=2000, completion_tokens=1000)

        assert cost is not None
        assert cost > 0

        # Restore original model
        config_module.settings.llm_model = original_model

    def test_cost_calculation_with_unknown_model(self, monkeypatch):
        from src.hbr_apple_rag import config as config_module

        original_model = config_module.settings.llm_model
        config_module.settings.llm_model = "unknown-model-xyz"

        engine = ComparisonEngine()
        cost = engine._calculate_cost(prompt_tokens=1000, completion_tokens=500)

        assert cost is None

        config_module.settings.llm_model = original_model