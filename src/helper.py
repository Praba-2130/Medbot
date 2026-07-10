"""
Helper Utilities for Medical Chatbot
====================================

Provides document loading, filtering, text splitting, and embedding
generation utilities for the RAG pipeline.
"""

import logging
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdf_file(data_dir: str) -> List[Document]:
    """
    Load all PDF files from the specified directory.

    Args:
        data_dir: Path to the directory containing PDF files.

    Returns:
        List of LangChain Document objects loaded from all PDFs.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        ValueError: If no PDF files are found in the directory.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {data_dir}")

    documents: List[Document] = []

    for pdf_file in pdf_files:
        logger.info("Loading PDF: %s", pdf_file.name)
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
        logger.info("Loaded %d pages from %s", len(docs), pdf_file.name)

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """
    Filter out empty or whitespace-only documents to reduce noise.

    Args:
        documents: List of Document objects.

    Returns:
        Filtered list containing only non-empty documents.
    """
    filtered: List[Document] = []

    for doc in documents:
        if doc.page_content and doc.page_content.strip():
            # Clean up excessive whitespace
            cleaned = " ".join(doc.page_content.split())
            doc.page_content = cleaned
            filtered.append(doc)

    logger.info(
        "Filtered %d documents down to %d non-empty documents",
        len(documents),
        len(filtered),
    )
    return filtered


def text_split(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter with medical-document-aware
    chunk sizes and overlap to preserve context.

    Args:
        documents: List of Document objects to split.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return HuggingFace sentence transformer embeddings.

    Uses the 'all-MiniLM-L6-v2' model which produces 384-dimensional
    dense vectors suitable for cosine similarity search.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info("Loaded HuggingFace embeddings model: %s", model_name)
    return embeddings