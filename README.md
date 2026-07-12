# Local PDF RAG

A local Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, FAISS, Sentence Transformers, and Ollama.

# Screenshots

### Main Interface

<p align="center">
  <img src="assets/home.png" width="800">
</p>

### Chat Example

<p align="center">
  <img src="assets/chat.png" width="800">
</p>

### Source Settings

<p align="center">
  <img src="assets/source.png" width="350">
</p>

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="850">
</p>

## Features

- Upload PDF documents
- Semantic search using FAISS
- Local LLM inference with Ollama
- Conversation memory
- Source page citations
- Streamlit web interface

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama
- PyPDF

## Run

```bash
pip install -r requirements.txt
streamlit run app.py


