"""RAG Ingestion Pipeline - Build and persist the vector store."""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.hbr_apple_rag.config import settings


def _load_and_split_documents(pdf_path: str | Path) -> List[Document]:
    """Load one PDF and split into chunks."""
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_documents(data_dir: Optional[Path] = None) -> Chroma:
    """Ingest all PDFs from data directory and persist the vector store."""
    data_dir = data_dir or Path(settings.data_dir)
    all_docs: List[Document] = []

    for pdf_file in sorted(data_dir.glob("*.pdf")):
        print(f"Processing: {pdf_file.name}")
        chunks = _load_and_split_documents(pdf_file)
        all_docs.extend(chunks)

    if not all_docs:
        raise ValueError(f"No PDF files found in {data_dir}")

    print(f"Total chunks created: {len(all_docs)}")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(settings.vector_store_dir),
    )
    print(f"Vector store persisted to: {settings.vector_store_dir}")
    return vector_store


if __name__ == "__main__":
    print("=== RAG Ingestion ===")
    print(f"Data dir: {settings.data_dir}")
    print(f"Vector store: {settings.vector_store_dir}\n")

    ingest_documents()
    print("\nIngestion completed successfully.")