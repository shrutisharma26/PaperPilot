
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample research paper chunks
texts = [
    "The Transformer architecture relies entirely on attention mechanisms.",
    "Transformers eliminate recurrence and allow parallel processing.",
    "The model uses multi-head self-attention.",
    "The paper introduces the Adam optimizer.",
    "The weather today is sunny and warm."
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for documents
embeddings = model.encode(texts)

embeddings = np.array(embeddings, dtype="float32")

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# Add embeddings to index
index.add(embeddings)

print("Total vectors indexed:", index.ntotal)

# Search query
query = "How does the Transformer use attention?"

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

faiss.normalize_L2(query_embedding)

# Search top 3 similar passages
scores, indices = index.search(query_embedding, k=3)

print("\nSearch results:\n")

for rank, idx in enumerate(indices[0]):
    print(f"Rank {rank + 1}")
    print("Score:", scores[0][rank])
    print("Text:", texts[idx])
    print()
