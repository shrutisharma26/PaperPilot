
from pdf_loader import extract_text
from chunking import chunk_text
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from pathlib import Path

# Load PDF
pages = extract_text("data/papers/attention.pdf")

# Create chunks with page metadata
all_chunks = []

for page in pages:
    chunks = chunk_text(page["text"])

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "page": page["page"]
        })

print("Total chunks:", len(all_chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(embeddings, dtype="float32")

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Indexed vectors:", index.ntotal)

# Create storage directory
Path("storage").mkdir(exist_ok=True)

# Save FAISS index
faiss.write_index(index, "storage/paper_index.faiss")

# Save chunk metadata
with open("storage/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("Index and chunks saved successfully!")
