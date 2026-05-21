"""RAG Query Engine - Load persisted store and answer questions."""

from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

from hbr_apple_rag.config import settings
from hbr_apple_rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


def _load_vector_store() -> Chroma:
    """Load existing persisted Chroma vector store."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )
    return Chroma(
        persist_directory=str(settings.vector_store_dir),
        embedding_function=embeddings,
    )


def _get_retriever(vector_store: Chroma):
    """Return configured retriever (MMR for better diversity)."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k},
    )


def generate_rag_response(question: str, vector_store: Optional[Chroma] = None) -> str:
    """Generate a grounded answer using the persisted RAG."""
    if vector_store is None:
        vector_store = _load_vector_store()

    retriever = _get_retriever(vector_store)
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=RAG_USER_TEMPLATE.format(context=context, question=question)),
    ]

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    response = llm.invoke(messages)
    return response.content.strip()