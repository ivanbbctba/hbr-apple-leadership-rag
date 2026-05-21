"""ResponseEngine - Handles Raw LLM, Prompt Engineering, and RAG."""

from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from hbr_apple_rag.config import settings
from hbr_apple_rag.prompts import (
    PROMPT_ENGINEERING_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)


class ResponseEngine:
    """
    Unified engine to generate responses in three modes:
        - "raw"        : Raw LLM (just user prompt, like notebook)
        - "prompt_eng" : Prompt Engineering (strong system prompt, no context)
        - "rag"        : RAG (same system prompt + retrieved context)
    """

    def __init__(self, mode: str = "rag"):
        self.mode = mode
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )
        self._retriever = None

        if self.mode == "rag":
            self._setup_retriever()

    def _setup_retriever(self):
        """Load existing persisted vector store."""
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )

        # Load from disk (already created by rag_pipeline.py)
        vector_store = Chroma(
            persist_directory=str(settings.vector_store_dir),
            embedding_function=embeddings,
        )

        self._retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.retriever_k},
        )

    def respond(self, question: str) -> str:
        if self.mode == "raw":
            return self._raw_llm_response(question)
        elif self.mode == "prompt_eng":
            return self._prompt_engineering_response(question)
        elif self.mode == "rag":
            return self._rag_response(question)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _raw_llm_response(self, question: str) -> str:
        """Raw LLM - same as my original jupyter notebook (just user prompt)."""
        messages = [HumanMessage(content=question)]
        response = self._llm.invoke(messages)
        return response.content.strip()

    def _prompt_engineering_response(self, question: str) -> str:
        """Prompt Engineering - uses the exact system prompt from my jupyter notebook."""
        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()

    def _rag_response(self, question: str) -> str:
        """RAG - same system prompt + retrieved context."""
        if self._retriever is None:
            raise ValueError("Retriever not initialized for RAG mode.")

        relevant_docs = self._retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=RAG_USER_TEMPLATE.format(context=context, question=question)),
        ]

        response = self._llm.invoke(messages)
        return response.content.strip()