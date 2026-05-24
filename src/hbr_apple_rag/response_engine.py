"""ResponseEngine - Unified class to generate responses using different strategies.

This class was created to allow easy comparison between three approaches:
    - Raw LLM
    - Prompt Engineering
    - RAG (Retrieval-Augmented Generation)

The main goal is to keep the comparison fair by using the same system prompt
for both Prompt Engineering and RAG modes. The only difference between them
is the presence of retrieved context from the document.

Design choice (kept as you prefer):
    One engine instance = one strategy (raw / prompt_eng / rag).
    This keeps the "step-by-step" visibility in the UI clear and explicit.

Heavy objects (retriever) are cached via lru_cache so creating many
ResponseEngine instances during comparisons stays fast.
"""

from functools import lru_cache
from typing import List

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
    """
    Load the persisted Chroma vector store once and reuse it.
    This makes repeated ResponseEngine(mode="rag") calls cheap after the first load.
    """
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
    """
    Generates responses in one of three modes.

    Modes:
        - "raw"        : Only the user question (baseline)
        - "prompt_eng" : Strong system prompt, no document context
        - "rag"        : Same system prompt + retrieved context

    The retriever is loaded through a cached helper so the class stays lightweight
    even when ComparisonEngine creates multiple instances.
    """

    def __init__(self, mode: str = "rag"):
        valid_modes = {"raw", "prompt_eng", "rag"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Valid options are: {', '.join(sorted(valid_modes))}"
            )

        self.mode = mode
        self._last_context: List = []

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
        """Return chunks retrieved in the last RAG call (empty for other modes)."""
        return self._last_context

    # --- Internal implementations ---

    def _raw_llm_response(self, question: str) -> str:
        """Baseline: send ONLY the user question. No system prompt at all."""
        # Passing a plain string is the cleanest way to guarantee
        # zero system prompt / hidden instructions for the baseline.
        response = self._llm.invoke(question)
        return response.content.strip()

    def _prompt_engineering_response(self, question: str) -> str:
        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()

    def _rag_response(self, question: str) -> str:
        if self._retriever is None:
            raise ValueError("Retriever was not initialized. Use mode='rag'.")

        relevant_docs = self._retriever.invoke(question)
        self._last_context = relevant_docs   # store for UI / ComparisonEngine

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=RAG_USER_TEMPLATE.format(context=context, question=question)),
        ]

        response = self._llm.invoke(messages)
        return response.content.strip()