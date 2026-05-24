from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.hbr_apple_rag.config import settings, MODEL_PRICING
from src.hbr_apple_rag.response_engine import ResponseEngine
from src.hbr_apple_rag.metrics import MetricsCollector
from src.hbr_apple_rag.evaluation import evaluate_faithfulness, evaluate_relevance


@dataclass
class ModeResult:
    answer: str
    response_time: float
    context: List = field(default_factory=list)
    error: Optional[str] = None

    # Token & Cost tracking
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

    # LLM-as-Judge scores
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None


class ComparisonEngine:
    def run_mode(self, mode: str, question: str) -> ModeResult:
        metrics = MetricsCollector()
        engine = ResponseEngine(mode=mode)

        metrics.start()
        answer = ""
        error = None
        try:
            answer = engine.respond(question)
        except Exception as exc:
            error = str(exc)
            answer = f"Sorry, an error occurred while generating the response.\n{error}"
        metrics.stop()

        # Get context (only relevant for RAG mode)
        context_list = engine.get_last_context() if mode == "rag" else []

        # Run evaluation
        faithfulness = None
        relevance = None
        if not error and answer:
            try:
                if mode == "rag" and context_list:
                    context_text = "\n\n".join([d.page_content for d in context_list])
                    faithfulness = evaluate_faithfulness(question, answer, context_text)
                relevance = evaluate_relevance(question, answer)
            except Exception:
                pass  # Evaluation failure should not break the whole flow

        # Token usage
        usage = engine.get_last_usage()
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens") or (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens or completion_tokens
            else None
        )

        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        return ModeResult(
            answer=answer,
            response_time=metrics.response_time or 0.0,
            context=context_list,  # ← Fixed: using context_list
            error=error,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            faithfulness_score=faithfulness,
            relevance_score=relevance,
        )

    def run_all(self, question: str) -> Dict[str, ModeResult]:
        return {
            "raw": self.run_mode("raw", question),
            "prompt_eng": self.run_mode("prompt_eng", question),
            "rag": self.run_mode("rag", question),
        }

    def _calculate_cost(
        self, prompt_tokens: Optional[int], completion_tokens: Optional[int]
    ) -> Optional[float]:
        if not prompt_tokens and not completion_tokens:
            return None

        model = settings.llm_model
        pricing = MODEL_PRICING.get(model)

        if not pricing:
            return None  # Unknown model or pricing not defined

        prompt_price = pricing.get("prompt", 0)
        completion_price = pricing.get("completion", 0)

        cost = (prompt_tokens or 0) * prompt_price + (completion_tokens or 0) * completion_price
        return round(cost, 6)