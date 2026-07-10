
# MedBot - AI Medical Assistant

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-6.x-purple)
![Cohere](https://img.shields.io/badge/Cohere-Command%20R%20Plus-red)

A production-ready AI-powered Medical Assistant built with **Retrieval-Augmented Generation (RAG)**. The chatbot answers medical questions based strictly on retrieved context from uploaded PDF documents, ensuring accurate and safe responses.

## Features

- **PDF Document Ingestion**: Automatically loads and processes medical PDFs from the `data/` folder
- **Intelligent Chunking**: Splits documents into context-preserving chunks using RecursiveCharacterTextSplitter
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` for efficient 384-dimensional dense vector generation
- **Pinecone Vector Store**: Serverless vector database for fast semantic similarity search
- **Cohere Command R Plus**: State-of-the-art LLM for generating accurate medical responses
- **LangChain RAG Pipeline**: Modern retrieval chain with proper context grounding
- **Modern Web UI**: Responsive, accessible chat interface with loading animations and mobile support
- **Safety-First Design**: System prompt enforces context-only answers and medical disclaimers
- **Production Logging**: Comprehensive logging instead of print statements
- **Type Hints**: Fully typed codebase following PEP 8 standards

## Installation

### Prerequisites

- Python 3.12 or higher
- Pinecone account (free tier available)
- Cohere API key

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd Medical-Chatbot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt