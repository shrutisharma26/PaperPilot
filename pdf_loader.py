
import pymupdf
from pathlib import Path

PDF_PATH = Path("data/papers/attention.pdf")


def extract_text(pdf_path):
    """Extract text from every page of a PDF."""

    document = pymupdf.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        extracted_pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return extracted_pages


if __name__ == "__main__":
    pages = extract_text(PDF_PATH)

    print(f"Total pages: {len(pages)}")

    print("\nFirst page preview:\n")
    print(pages[0]["text"][:1500])
