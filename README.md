# 🩺 Medical Assistant Chatbot 💊

A **Medical Question-Answering Chatbot** using **Retrieval-Augmented Generation (RAG)** powered by **FAISS**, **LangChain**, **Hugging Face Embeddings**, **Groq's LLaMA-3**. It provides concise, context-aware answers to medical questions based on PDF documents.

---

## 💡 Overview

This project leverages domain-specific documents (e.g., medical papers, guidelines) to generate accurate and relevant answers to user queries. It combines:

- **Document retrieval** via FAISS and Hugging Face embeddings
- **Prompted LLM responses** using Groq-hosted LLaMA-3
- **Streamlit frontend** for interaction

---

## 🔍 How It Works

1. **PDF Loading**: All PDFs in the `data/` folder are loaded and preprocessed.
2. **Text Splitting**: Documents are split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**: Text chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Store**: Chunks are stored in a FAISS index for semantic search.
5. **RAG Pipeline**:
   - Relevant chunks are retrieved based on user input.
   - Combined with a custom prompt and passed to a Groq-hosted LLaMA-3 model.
   - A concise, medically-informed answer is returned to the user.
6. **Frontend**: The chatbot interface is built using Streamlit with support for user input, loading spinners, and clean output formatting.
