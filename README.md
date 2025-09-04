# 🎬 Semantic Netflix Movie Recommender

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **semantic movie recommendation system** using **BGE-M3 embeddings** and a **Cross-Encoder reranker**, providing highly relevant Netflix movie suggestions. Built with **LangChain, ChromaDB, and Gradio**. Poster images are loaded directly from the CSV metadata URLs.

---

## 🌟 Features

- **Semantic search** – find movies based on a description instead of exact keywords.  
- **State-of-the-art embeddings** – powered by `BAAI/bge-m3`.  
- **Reranking** – Cross-Encoder ensures the most relevant movies appear at the top.  
- **Poster display** – loads posters directly from CSV URLs.  
- **Offline-ready vector database** – prebuilt Chroma index for fast local search.  
- **Interactive UI** – Gradio web interface for easy browsing.

---

## 🗂️ Project Structure
project/
│─ app.py # Main application
│─ chroma_movies_bge/ # Prebuilt Chroma vector database
│─ FINALE_NEW_WITH_IMAGES.csv # Movie metadata (titles, descriptions, poster URLs)
│─ placeholder.png # Placeholder image if a poster is missing
│─ requirements.txt # Python dependencies


---

## 🚀 Usage
python app.py


The Gradio interface will open in your browser.

Enter a description of your ideal movie (e.g., "A sci-fi movie about time travel").

Click Find recommendations to browse the top recommended movies with posters.

---


## 🧠 How It Works

Chroma Vector Database – stores embeddings of movie descriptions locally.

Semantic Search – finds the 50 most similar movies to your query.

Cross-Encoder Reranking – reranks results for maximum relevance.

Gallery Display – top recommendations are shown with poster images in the Gradio UI.

--- 

## ✅ Notes

Requires internet access to fetch poster images.

chroma_movies_bge folder contains the prebuilt embeddings – no retraining necessary.

Compatible with Windows, macOS, and Linux (Python 3.10+ recommended).

---

## 📦 Dependencies

Python 3.10+
pandas
sentence-transformers
langchain
langchain-chroma
chromadb
gradio
torch

---

## 📷 Optional

If a poster URL is missing or broken, placeholder.png will be shown instead.

---

## ⚡ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <project-folder>

# Create a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


🔗 License

MIT License


---

