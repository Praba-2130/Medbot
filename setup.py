"""
Medical Chatbot - Setup Configuration
=====================================

Package setup for the Medical Chatbot RAG application.
"""

from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Lakshmi Praba",
    author_email="lakshmipraba.2113@gmail.com",
    description="AI-powered Medical Assistant using RAG with Flask, LangChain, Pinecone, and Cohere",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lakshmipraba/medical-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "flask>=3.0.0",
        "python-dotenv>=1.0.0",
        "pypdf>=4.0.0",
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-cohere>=0.3.0",
        "langchain-pinecone>=0.2.0",
        "langchain-huggingface>=0.1.0",
        "pinecone>=6.0.0",
        "sentence-transformers>=3.0.0",
        "cohere>=5.0.0",
    ],
)