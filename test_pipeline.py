
from pdf_loader import extract_text
from chunking import chunk_text

pages = extract_text("data/papers/attention.pdf")

all_chunks = []

for page in pages:

    chunks = chunk_text(page["text"])

    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "page": page["page"]
        })

print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))

print("\nFirst chunk:")
print(all_chunks[0])
