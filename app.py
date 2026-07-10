#!/usr/bin/env python3
"""
Medical Chatbot - Flask Application
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import logging
import os

from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hardcoded API keys
PINECONE_API_KEY = "pcsk_vR7e2_4qwRU1jXbEGRwTEbcniSywaXo3DAmoQmmvnkogszd6S3facLXpJxmwKsJJRXfzY"
COHERE_API_KEY = "cohere_m37jaTf7J2s8xU8Qo3INo2jDoBcdZvgKLL4WozQF2po43Q"
PINECONE_INDEX_NAME = "medical-chatbot"

# Set environment variables for libraries that need them
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

logger.info("Loading embeddings...")
embeddings = download_hugging_face_embeddings()

logger.info("Connecting to Pinecone index: %s", PINECONE_INDEX_NAME)
try:
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
except Exception as e:
    logger.error("Failed to connect to Pinecone: %s", e)
    sys.exit(1)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

logger.info("Initializing Cohere LLM...")
llm = ChatCohere(
    model="command-r-plus-08-2024",
    temperature=0.4,
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Build RAG chain using LCEL
rag_chain = (
    RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

logger.info("RAG chain ready. Starting server...")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg", "").strip()

    if not user_input:
        return "Please enter a valid question."

    logger.info("User query: %s", user_input)

    try:
        answer = rag_chain.invoke(user_input)
        return answer
    except Exception as e:
        logger.error("Error processing request: %s", e, exc_info=True)
        return "Error: " + str(e)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )