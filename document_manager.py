
import streamlit as st
import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


@st.cache_data
def extract_and_embed_pdf(pdf_bytes):
    """Extract text and generate embeddings for an uploaded PDF."""

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunks = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        chunk_size = 500
        overlap = 100
        start = 0

        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            start += chunk_size - overlap

    document.close()

    if not chunks:
        raise ValueError("No readable text found in this PDF.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    ).astype("float32")

    return chunks, embeddings


def process_pdf(pdf_bytes):
    chunks, embeddings = extract_and_embed_pdf(pdf_bytes)

    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks