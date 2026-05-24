"""ResponseEngine - Unified class to generate responses using different strategies.

Design choice kept as you prefer:
    One engine instance = one strategy (raw / prompt_eng / rag).
    This keeps the "step-by-step" visibility clear.
"""

from functools import lru_cache
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from src.hbr_apple_rag.config import settings
from src.hbr_apple_rag.prompts import (
    PROMPT_ENGINEERING_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)


@lru_cache(maxsize=1)
def get_retriever():
    """Cached retriever so creating multiple ResponseEngine instances is cheap."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    vector_store = Chroma(
        persist_directory=str(settings.vector_store_dir),
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": settings.retriever_lambda_mult,
        },
    )


class ResponseEngine:
    def __init__(self, mode: str = "rag"):
        valid_modes = {"raw", "prompt_eng", "rag"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'")

        self.mode = mode
        self._last_context: List = []
        self._last_usage: Dict[str, Any] = {}

        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )

        self._retriever = None
        if self.mode == "rag":
            self._retriever = get_retriever()

    def respond(self, question: str) -> str:
        if self.mode == "raw":
            return self._raw_llm_response(question)
        elif self.mode == "prompt_eng":
            return self._prompt_engineering_response(question)
        elif self.mode == "rag":
            return self._rag_response(question)
        else:
            raise ValueError(f"Invalid mode '{self.mode}'")

    def get_last_context(self) -> List:
        return self._last_context

    def get_last_usage(self) -> Dict[str, Any]:
        """Return token usage from the last LLM call."""
        return self._last_usage

    # --- Internal methods ---

    def _raw_llm_response(self, question: str) -> str:
        response = self._llm.invoke(question)
        self._last_usage = getattr(response, "usage_metadata", {}) or {}
        return response.content.strip()

    def _prompt_engineering_response(self, question: str) -> str:
        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        self._last_usage = getattr(response, "usage_metadata", {}) or {}
        return response.content.strip()

    def _rag_response(self, question: str) -> str:
        if self._retriever is None:
            raise ValueError("Retriever was not initialized. Use mode='rag'.")

        relevant_docs = self._retriever.invoke(question)
        self._last_context = relevant_docs

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=RAG_USER_TEMPLATE.format(context=context, question=question)),
        ]

        response = self._llm.invoke(messages)
        self._last_usage = getattr(response, "usage_metadata", {}) or {}
        return response.content.strip()