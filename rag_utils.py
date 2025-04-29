import fitz  # PyMuPDF
import re  # Regular expressions
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once to avoid reloading on every call
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text_preserve_paragraphs(text: str) -> str:
    """
    Cleans the input text while preserving paragraph structure.

    This function removes multiple consecutive newlines and replaces them with a single newline.
    It also replaces multiple spaces or tabs with a single space.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with preserved paragraph structure.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_text_from_pdfs(pdf_files: List[bytes]) -> str:
    """
    Extracts text from a list of PDF files.

    Args:
        pdf_files (List[bytes]): A list of PDF file contents in bytes.

    Returns:
        str: The extracted text from all the PDF files concatenated into a single string.
    """
    full_text = ""
    for content in pdf_files:
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            full_text += page.get_text()
        doc.close()
    # Clean the text to preserve paragraphs
    full_text = clean_text_preserve_paragraphs(full_text)
    return full_text


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits a large text into smaller chunks of a specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The size of each chunk. Defaults to 500.

    Returns:
        List[str]: A list of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): A list of text chunks to be embedded.

    Returns:
        np.ndarray: A NumPy array containing the embeddings for the text chunks.
    """
    return embedder.encode(chunks, convert_to_numpy=True)


def embed_query(question: str) -> np.ndarray:
    """
    Generates an embedding for a single query string.

    Args:
        question (str): The query string to be embedded.

    Returns:
        np.ndarray: A NumPy array representing the embedding of the query.
    """
    return embedder.encode([question], convert_to_numpy=True)[0]
