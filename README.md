# README: Local Deployment for RAG Researcher

This document contains complete instructions to run the **RAG Researcher** project locally. The application allows users to upload PDF files, generate text embeddings, and retrieve contextually relevant document chunks based on user-input questions. The output is optimized to be copy-pasted into a large language model like ChatGPT.

---

## ✅ Requirements

Before running the project, make sure the following are installed:

- Python 3.8 or newer
- pip (Python package manager)

### Python Dependencies
All required Python libraries are listed in `requirements.txt`. These include:
```
fastapi
uvicorn
python-multipart
PyMuPDF
sentence-transformers
numpy
```

To install them, run:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure
```
RAG-Researcher/
├── app.py                  # FastAPI backend
├── rag_utils.py            # Text extraction, chunking, and embedding logic
├── requirements.txt        # Python dependencies
└── static/
    ├── index.html          # Frontend HTML page
    ├── style.css           # Optional styling file
    └── rag.png             # Logo/icon used in the UI
```

---

## 🚀 How to Run the App Locally

1. **Clone the Repository or Download the Files**
2. **Open a Terminal in the project folder**
3. **Run the FastAPI app using Uvicorn:**
```bash
uvicorn app:app --host=0.0.0.0 --port=8000
```
4. **Open your browser and navigate to:**
```
http://localhost:8000
```
You’ll see the RAG Researcher interface.

---

## 💻 How to Use the Application

1. **Upload one or more PDFs** using the file input.
2. Click **"Create Embeddings"** to process and embed the text.
3. Enter a **question** related to the PDFs.
4. Select the number of chunks to retrieve and whether to include full prompt formatting.
5. Click **"Generate Text"** to receive the most relevant chunks or a copy-ready prompt.
6. Use the **"Copy Text"** button to copy the result into ChatGPT or another LLM interface.

---

## 🧠 Notes
- This project is designed for local use. It does not rely on external APIs or require any API keys.
- The embedding model used is `all-MiniLM-L6-v2` from `sentence-transformers`, which works offline.
- The application does not store files or logs any user input.

