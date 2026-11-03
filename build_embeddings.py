
from pdf_loader import extract_text
from chunking import chunk_text
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load the research paper
pages = extract_text("data/papers/attention.pdf")

# Step 2: Generate chunks
all_chunks = []

for page in pages:
    chunks = chunk_text(page["text"])

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "page": page["page"]
        })

print("Total chunks:", len(all_chunks))

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Extract chunk text
texts = [chunk["text"] for chunk in all_chunks]

# Step 5: Generate embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True
)

# Step 6: Convert to NumPy array
embeddings = np.array(embeddings, dtype="float32")

print("Embedding shape:", embeddings.shape)
print("Embedding type:", embeddings.dtype)
