#!/usr/bin/env python3
"""
Vector Store Index Builder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import logging
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Hardcoded API keys
PINECONE_API_KEY = "pcsk_vR7e2_4qwRU1jXbEGRwTEbcniSywaXo3DAmoQmmvnkogszd6S3facLXpJxmwKsJJRXfzY"
PINECONE_INDEX_NAME = "medical-chatbot"

# Set environment variable
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load Documents
try:
    logger.info("Loading PDF documents from data/...")
    documents = load_pdf_file("data/")
    documents = filter_to_minimal_docs(documents)
    text_chunks = text_split(documents)
except FileNotFoundError as e:
    logger.error("Data directory error: %s", e)
    sys.exit(1)
except ValueError as e:
    logger.error("Document loading error: %s", e)
    sys.exit(1)

# Embeddings
logger.info("Loading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()

# Pinecone
logger.info("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete old index if exists (to fix dimension mismatch)
if pc.has_index(PINECONE_INDEX_NAME):
    logger.info("Deleting old index '%s' to recreate with correct dimension...", PINECONE_INDEX_NAME)
    pc.delete_index(PINECONE_INDEX_NAME)
    logger.info("Old index deleted.")

# Create Index with correct dimension (384 for all-MiniLM-L6-v2)
logger.info("Creating index '%s' with dimension 384...", PINECONE_INDEX_NAME)
pc.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    deletion_protection="disabled",
)
logger.info("Index created successfully.")

# Upload Embeddings
try:
    logger.info("Uploading %d chunks to Pinecone...", len(text_chunks))
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    logger.info("Vector Database Created Successfully!")
except Exception as e:
    logger.error("Failed to upload embeddings: %s", e)
    sys.exit(1)