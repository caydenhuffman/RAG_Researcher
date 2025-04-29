from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import numpy as np

from rag_utils import extract_text_from_pdfs, chunk_text, embed_chunks, embed_query

app = FastAPI()

# Globals to store session data
stored_chunks: List[str] = []
stored_embeddings: np.ndarray = None

# Allow frontend JS to talk to backend if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> FileResponse:
    """
    Serves the frontend HTML page.
    Endpoint: GET /
    Returns:
        FileResponse: Serves the static/index.html file.
    Purpose:
        When a user navigates to localhost:8000, this serves the frontend.
    """
    return FileResponse("static/index.html")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/upload_pdfs")
async def upload_pdfs(pdfs: List[UploadFile] = File(...)) -> dict:
    """
    Handles PDF upload, extracts text, chunks it, and creates embeddings.
    Endpoint: POST /upload_pdfs
    Args:
        pdfs (List[UploadFile]): List of uploaded PDF files.
    Returns:
        dict: A success message.
    Purpose:
        When the "Create Embeddings" button is clicked, PDFs are sent here,
        processed into chunks, and embeddings are created and stored.
    """
    global stored_chunks, stored_embeddings

    pdf_contents = [await pdf.read() for pdf in pdfs]
    full_text = extract_text_from_pdfs(pdf_contents)
    stored_chunks = chunk_text(full_text)
    stored_embeddings = embed_chunks(stored_chunks)

    return {"message": "Embeddings created successfully."}

@app.post("/query")
async def query(
    question: str = Form(...),
    chunk_count: int = Form(...),
    full_prompt: str = Form(...)
) -> dict:
    """
    Accepts a question, finds the most relevant chunks, and returns them with structured formatting.
    Endpoint: POST /query
    Args:
        question (str): The user's input question.
        chunk_count (int): Number of chunks to return.
        full_prompt (str): Whether to wrap results in a full prompt or just chunks.
    Returns:
        dict: The generated formatted text.
    Purpose:
        When the "Generate Text" button is clicked, the question is sent here,
        embedded, and compared to stored chunks to find the best matches, including similarity scores.
    """
    global stored_chunks, stored_embeddings

    if stored_embeddings is None or len(stored_chunks) == 0:
        return {"generated_text": "Error: No embeddings found. Please upload PDFs first."}

    question_embedding = embed_query(question)

    # Compute cosine similarities
    similarities = stored_embeddings @ question_embedding / (
        np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )

    # Get top-k chunks
    top_indices = np.argsort(similarities)[-int(chunk_count):][::-1]
    top_chunks = [stored_chunks[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    # Intro prompt template
    prompt_intro = (
        "You are an AI assistant. You will be given a question inside <Question> tags.\n"
        "You will also be given lecture notes inside <LectureNote> tags.\n"
        "Please answer the question as accurately as possible.\n"
        "You may use the lecture notes to help you answer the question.\n"
        "But answer the question to the best of your ability and use the tools available to you.\n"
        "Aim for conciseness and clarity.\n"
    )

    # Assemble tags
    question_tag = f"<Question>\n{question}\n</Question>"
    formatted_chunks = []
    for idx, (chunk, score) in enumerate(zip(top_chunks, top_scores)):
        if full_prompt.lower() == "true":
            formatted_chunks.append(f"<LectureNote>\n{chunk}\n</LectureNote>")
        else:
            formatted_chunks.append(f"<LectureNote similarity={score:.4f}>\n{chunk}\n</LectureNote>")

    context = "\n\n".join(formatted_chunks)

    generated_text = f"{prompt_intro}\n\n{question_tag}\n\n{context}"

    return {"generated_text": generated_text}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)