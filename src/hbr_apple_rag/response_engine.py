"""ResponseEngine - Unified class to generate responses using different strategies.

This class was created to allow easy comparison between three approaches:
    - Raw LLM
    - Prompt Engineering
    - RAG (Retrieval-Augmented Generation)

The main goal is to keep the comparison fair by using the same system prompt
for both Prompt Engineering and RAG modes. The only difference between them
is the presence of retrieved context from the document.
"""

from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from src.hbr_apple_rag.config import settings
from src.hbr_apple_rag.prompts import (
    PROMPT_ENGINEERING_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)


class ResponseEngine:
    """
    A simple but clean class to generate responses in three different modes.

    Modes available:
        - "raw"        : Sends only the user question to the LLM (baseline).
        - "prompt_eng" : Uses a strong system prompt (no document context).
        - "rag"        : Uses the same system prompt + relevant context from the PDF.

    This design allows us to clearly demonstrate the value of good prompting
    and the additional benefit of retrieval (RAG) in a controlled way.

    Example usage:
        engine = ResponseEngine(mode="rag")
        answer = engine.respond("Who are the authors of the article?")
    """

    def __init__(self, mode: str = "rag"):
        """
        Initialize the ResponseEngine.

        Args:
            mode: The response strategy to use. Options are "raw", "prompt_eng", or "rag".
                  Default is "rag".
        """
        valid_modes = {"raw", "prompt_eng", "rag"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Valid options are: {', '.join(sorted(valid_modes))}"
            )
        
        self.mode = mode
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )
        self._retriever: Optional[Chroma] = None

        # Only initialize the retriever when using RAG mode.
        # This avoids unnecessary loading when we only need raw or prompt engineering.
        if self.mode == "rag":
            self._load_retriever()

    def _load_retriever(self) -> None:
        """
        Load the existing persisted vector store from disk.

        Note: We only load here. Document ingestion/creation is handled
        separately in rag_pipeline.py to keep concerns separated.
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

        self._retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.retriever_k},
        )

    def respond(self, question: str) -> str:
        """
        Generate a response based on the selected mode.

        Args:
            question: The question to answer.

        Returns:
            The generated answer as a string.
        """
        if self.mode == "raw":
            return self._raw_llm_response(question)
        elif self.mode == "prompt_eng":
            return self._prompt_engineering_response(question)
        elif self.mode == "rag":
            return self._rag_response(question)
        else:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                "Valid options are: 'raw', 'prompt_eng', 'rag'"
            )

    def _raw_llm_response(self, question: str) -> str:
        """
        Raw LLM mode - sends only the user question (same behavior as the notebook).
        No system prompt is used.
        """
        messages = [HumanMessage(content=question)]
        response = self._llm.invoke(messages)
        return response.content.strip()

    def _prompt_engineering_response(self, question: str) -> str:
        """
        Prompt Engineering mode - uses a strong system prompt but no document context.
        This follows the same system prompt used in the original notebook.
        """
        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()

    def _rag_response(self, question: str) -> str:
        """
        RAG mode - uses the same system prompt as Prompt Engineering + retrieved context.
        This allows a fair comparison between the two approaches.
        """
        if self._retriever is None:
            raise ValueError("Retriever was not initialized. Use mode='rag'.")

        relevant_docs = self._retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            SystemMessage(content=PROMPT_ENGINEERING_SYSTEM_PROMPT),
            HumanMessage(content=RAG_USER_TEMPLATE.format(context=context, question=question)),
        ]

        response = self._llm.invoke(messages)
        return response.content.strip()