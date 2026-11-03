
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into chunks with overlapping context.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    The Transformer architecture relies entirely on attention mechanisms.
    It removes recurrence and convolution, allowing parallel processing.
    This improves training efficiency and enables modeling long-range dependencies.
    """

    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)
