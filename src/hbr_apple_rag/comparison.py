from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.hbr_apple_rag.response_engine import ResponseEngine
from src.hbr_apple_rag.metrics import MetricsCollector


@dataclass
class ModeResult:
    """
    Structured result for a single execution mode.

    Attributes:
        answer: The text response from the LLM.
        response_time: Time taken in seconds.
        context: List of retrieved document chunks (only populated for RAG).
        error: Error message if something failed (None on success).
    """
    answer: str
    response_time: float
    context: List = field(default_factory=list)
    error: Optional[str] = None


class ComparisonEngine:
    """
    Orchestrates the three modes while collecting metrics.

    Keeps the explicit per-mode engine creation so the UI can show
    each strategy independently (the "step-by-step" educational feel).
    """

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
            answer = (
                "Sorry, an error occurred while generating the response.\n"
                f"Details: {error}"
            )
        metrics.stop()

        context = engine.get_last_context() if mode == "rag" else []

        return ModeResult(
            answer=answer,
            response_time=metrics.response_time or 0.0,
            context=context,
            error=error,
        )

    def run_all(self, question: str) -> Dict[str, ModeResult]:
        return {
            "raw": self.run_mode("raw", question),
            "prompt_eng": self.run_mode("prompt_eng", question),
            "rag": self.run_mode("rag", question),
        }