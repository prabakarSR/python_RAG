# 📄 Simple RAG Project – Ask Questions from Your TXT File

This is a simple Retrieval-Augmented Generation (RAG) system built using Python and ChromaDB.

The project allows you to upload a `.txt` file and ask questions based on its content.

---

## 🚀 What This Project Does

- Reads a `.txt` file (example: iPhone information)
- Splits text into chunks
- Converts text into vector embeddings
- Stores embeddings in ChromaDB
- Retrieves relevant content based on user query
- Generates answers using an LLM

---

## 📂 Project Files

embed_to_chroma.py # Converts TXT file into embeddings  
query_chroma.py # Searches similar content from ChromaDB  
rag_chat.py # Chat-based Q&A using RAG  
chroma_db/ # Vector database (ignored in Git)