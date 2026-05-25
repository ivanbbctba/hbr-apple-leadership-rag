"""LLM-as-Judge evaluation module."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.hbr_apple_rag.config import settings
from src.hbr_apple_rag.prompts import (
    FAITHFULNESS_JUDGE_PROMPT,
    RELEVANCE_JUDGE_PROMPT,
)


def _get_judge_llm():
    return ChatOpenAI(
        model=settings.judge_model,
        temperature=0.0,
        max_tokens=10,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )


def evaluate_faithfulness(question: str, answer: str, context: str) -> float:
    """Returns faithfulness score from 0 to 5."""
    llm = _get_judge_llm()
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        context=context, question=question, answer=answer
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0


def evaluate_relevance(question: str, answer: str) -> float:
    """Returns relevance score from 0 to 5."""
    llm = _get_judge_llm()
    prompt = RELEVANCE_JUDGE_PROMPT.format(question=question, answer=answer)
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except ValueError:
        return 0.0