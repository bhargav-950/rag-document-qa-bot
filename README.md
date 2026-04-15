# 📚 RAG Document Q&A Bot

## 🔍 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions on a collection of documents and receive accurate, context-based answers with source citations.

The system retrieves relevant document chunks using vector similarity search and generates answers using a Large Language Model (LLM).

---

## ⚙️ Tech Stack
- Python 3.11
- LangChain
- ChromaDB (Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq LLM (Llama 3.1)
- python-dotenv

---

## 🧠 Architecture (RAG Pipeline)
1. Document Ingestion – Load PDF, TXT, and DOCX files  
2. Text Chunking – Split documents into chunks (size: 700, overlap: 100)  
3. Embedding Generation – Convert chunks into vector embeddings  
4. Vector Storage – Store embeddings in ChromaDB  
5. Retrieval – Fetch top-k relevant chunks  
6. Answer Generation – Generate answer using LLM based on context  

---

## ✂️ Chunking Strategy
Used RecursiveCharacterTextSplitter with:
- Chunk Size: 700  
- Chunk Overlap: 100  

This helps maintain context across chunks and improves retrieval accuracy.

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your_repo_url>
cd rag-document-qa-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file and add:
```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ How to Run

### Step 1: Index Documents
```bash
python ingest.py
```

### Step 2: Run Q&A Bot
```bash
python app.py
```

---

## 💡 Example Queries
- What is machine learning?
- What is deep learning?
- Explain supervised learning
- What is clustering?
- What is overfitting?

---

## ⚠️ Limitations
- Cannot answer questions outside the provided documents  
- Performance depends on document quality  
- Retrieval may sometimes return less relevant chunks  

---

## 🎯 Features
- Context-based answers (no hallucination)
- Source citations (file name + page number)
- Retrieved chunk display
- Persistent vector database
- CLI-based interactive bot

---

## 📁 Project Structure
```
rag-document-qa-bot/
├── data/              # Input documents
├── vectorstore/       # Stored embeddings
├── ingest.py          # Indexing script
├── app.py             # Q&A bot
├── requirements.txt
├── README.md
```

---

## 🚀 Future Improvements
- Add Streamlit UI  
- Improve retrieval accuracy  
- Add reranking  
- Support more file formats  


**Note:** The `vectorstore/` folder is not included in this repository due to size limitations. Please run `python ingest.py` to generate embeddings before running the application.