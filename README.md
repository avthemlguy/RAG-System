RAG System – Document Retrieval & Semantic Search

A lightweight Retrieval-Augmented Generation (RAG) system built using:

Python

FAISS for vector search

Groq LLMs for fast embeddings & generation

Custom PDF data loader

Simple modular architecture (src/ folder)

This project loads PDF documents, converts them into vector embeddings, stores them using FAISS, and performs semantic search + summarization on top of the retrieved data.


🚀 Features
✔ PDF Ingestion

Extracts text from PDF files and splits them into chunks.

✔ Embedding Generation (Groq)

Uses Groq models to generate dense vector embeddings.

✔ FAISS Vector Store

Stores all embeddings locally inside faiss_store/.

✔ Semantic Search

Retrieves the top-k most relevant chunks from your corpus.


RAG-System/
│── main.py
│── requirements.txt
│── pyproject.toml
│── .gitignore
│── README.md
│── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── search.py
│   └── vectorstore.py
│── data/               # (ignored) PDF files
│── faiss_store/        # (ignored) FAISS index + meta
│── .env                # (ignored) environment variables


🔧 Installation

1️⃣ Clone the repository
git clone https://github.com/avthemlguy/RAG-System.git
cd RAG-System

2️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

✔ RAG-style Answer Generation

Combines context + query → produces an answer using Groq LLM.
