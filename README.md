# 🩺 Medical Assistant Chatbot 💊

A **Medical Question-Answering Chatbot** using **Retrieval-Augmented Generation (RAG)**. This chatbot answers user questions by retrieving information from trusted medical documents, which are processed into vector representations and stored in a vector database for efficient semantic search. It generates concise responses through a simple chat interface, making medical knowledge more accessible and easier to understand.

---

## 🔍 Overview

- Implemented a **Retrieval-Augmented Generation (RAG)** architecture for grounded responses.
- Uses a collection of trusted medical documents as the **Knowledge Base**.
- Converts documents into **Vector Embeddings** for semantic understanding.
- Stores processed data in a **Vector Database** to enable fast and relevant retrieval.
- Accepts natural language queries from the user via a **Prompt Interface**.
- Retrieves the most relevant document **Chunks** based on semantic similarity.
- Passes context to a **Large Language Model** to generate concise, accurate answers.
- Designed for quick access to **Medical** knowledge in an interactive format.
---

## Tech Stack

### Programming Language & Frameworks
- **Python** — Primary programming language for backend and scripting.
- **LangChain** — Orchestration framework for managing language model workflows and retrieval pipelines.
- **Streamlit** — Framework for building interactive web interface.

### Vector Database & Embeddings
- **FAISS (Facebook AI Similarity Search)** — Vector store indexing embeddings for efficient similarity search, enabling fast retrieval of relevant document chunks.
- **HuggingFace Embeddings (all-MiniLM-L6-v2)** — Pretrained transformer-based model converting text into dense vector representations capturing semantic meaning.

### Large Language Model
- **LLaMA-3** — High-capacity large language model generating context-aware, concise answers based on retrieved document context.
