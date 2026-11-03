
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("storage/paper_index.faiss")

# Load stored chunks
with open("storage/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def search_paper(query, top_k=3):

    # Convert question into embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Normalize embedding
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":

    while True:

        query = input("\nAsk a question (or type exit): ")

        if query.lower() == "exit":
            break

        results = search_paper(query)

        print("\nRelevant passages:\n")

        for result in results:

            print(f"Page: {result['page']}")
            print(f"Similarity: {result['score']:.4f}")
            print(result["text"])
            print("-" * 60)
