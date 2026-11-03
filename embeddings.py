
from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example research paper chunks
texts = [
    "The Transformer architecture uses attention mechanisms.",
    "Transformers rely on self-attention instead of recurrence.",
    "The weather is sunny today."
]

# Convert text into numerical vectors
embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])
