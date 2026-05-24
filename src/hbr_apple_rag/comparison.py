from typing import Dict, List, Any
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
        context: List of retrieved document chunks (only for RAG).
    """
    answer: str
    response_time: float
    context: List[Any] = field(default_factory=list)


class ComparisonEngine:
    """
    Orchestrates the execution of different RAG strategies and collects performance metrics.
    """

    def run_mode(self, mode: str, question: str) -> ModeResult:
        """
        Runs a single mode and returns its structured result.
        
        Args:
            mode: The mode to run ('raw', 'prompt_eng', 'rag').
            question: The user question.
            
        Returns:
            A ModeResult object.
        """
        metrics = MetricsCollector()
        engine = ResponseEngine(mode=mode)
        
        metrics.start()
        answer = engine.respond(question)
        metrics.stop()
        
        context = []
        if mode == "rag" and engine._retriever:
            # Note: We retrieve context again here to match the current app.py behavior.
            # In a future optimization, ResponseEngine could return this metadata.
            context = engine._retriever.invoke(question)
            
        return ModeResult(
            answer=answer,
            response_time=metrics.response_time,
            context=context
        )

    def run_all(self, question: str) -> Dict[str, ModeResult]:
        """
        Runs all three comparison modes for a given question.
        
        Args:
            question: The user question.
            
        Returns:
            A dictionary mapping mode names to ModeResult objects.
        """
        return {
            "raw": self.run_mode("raw", question),
            "prompt_eng": self.run_mode("prompt_eng", question),
            "rag": self.run_mode("rag", question),
        }
