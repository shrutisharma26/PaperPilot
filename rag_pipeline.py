
import re
import pickle
from pathlib import Path

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer


MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SAVED_INDEX_PATH = Path("storage/paper_index.faiss")
SAVED_CHUNKS_PATH = Path("storage/chunks.pkl")

TOP_K = 4

# CLI conversation history
conversation_history = []

# Load embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def _load_saved_retrieval_system():
    """Load the saved FAISS index and chunks for CLI use."""

    if not SAVED_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {SAVED_INDEX_PATH}. "
            "Process a PDF in PaperPilot first."
        )

    if not SAVED_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Saved chunks not found: {SAVED_CHUNKS_PATH}. "
            "Process a PDF in PaperPilot first."
        )

    saved_index = faiss.read_index(str(SAVED_INDEX_PATH))

    with SAVED_CHUNKS_PATH.open("rb") as file:
        saved_chunks = pickle.load(file)

    return saved_index, saved_chunks


def _format_context(retrieved_chunks):
    """Format chunks and collect valid page numbers."""

    context_parts = []
    allowed_pages = set()

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        page = chunk.get("page")
        text = str(chunk.get("text", "")).strip()

        if page is None or not text:
            continue

        page = str(page)
        allowed_pages.add(page)

        context_parts.append(f"[Page {page}]\n{text}")

    return "\n\n".join(context_parts), allowed_pages


def _remove_invalid_page_citations(answer, allowed_pages):
    """Remove citations to pages not present in retrieved context."""

    def replace(match):
        page = match.group(1)

        if page in allowed_pages:
            return match.group(0)

        return ""

    return re.sub(
        r"\[Page\s+(\d+)\]",
        replace,
        answer,
        flags=re.IGNORECASE,
    )


def _retrieve(index, chunks, question, top_k=TOP_K):
    """Retrieve the most relevant chunks."""

    if index is None or chunks is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32,
    )

    faiss.normalize_L2(question_embedding)

    _, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []

    for idx in indices[0]:
        if idx != -1 and 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


def retrieve_context(question, top_k=3):
    """Retrieve chunks from the saved paper index."""

    saved_index, saved_chunks = _load_saved_retrieval_system()

    return _retrieve(
        saved_index,
        saved_chunks,
        question,
        top_k=top_k,
    )


def _build_messages(question, context, allowed_pages, history=None):
    """Build the prompt and Ollama message history."""

    allowed_page_text = ", ".join(
        sorted(allowed_pages, key=lambda page: int(page))
    )

    prompt = f"""You are PaperPilot, a research-paper assistant.

Answer using only the supplied document excerpts.

Rules:
- Be clear, concise, and directly answer the question.
- Cite factual claims using [Page X].
- You may cite ONLY these page numbers: {allowed_page_text}.
- Cite a page only if its excerpt supports the claim.
- Never invent page numbers, quotations, findings, or details.
- If the excerpts do not answer the question, say so.
- Use conversation history only to understand follow-up questions.
- Do not use outside knowledge to fill gaps.

Document excerpts:
{context}

Question:
{question}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are PaperPilot, a careful research assistant. "
                "Ground factual answers in retrieved document excerpts."
            ),
        }
    ]

    if history:
        for message in history:
            if (
                isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            ):
                messages.append({
                    "role": message["role"],
                    "content": message["content"],
                })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    return messages


def _generate_answer(question, index, chunks, history=None, top_k=TOP_K):
    """Retrieve context and generate an answer using Ollama."""

    retrieved_chunks = _retrieve(
        index,
        chunks,
        question,
        top_k=top_k,
    )

    context, allowed_pages = _format_context(retrieved_chunks)

    if not context:
        return (
            "I couldn't find relevant excerpts in this document "
            "to answer that."
        )

    messages = _build_messages(
        question=question,
        context=context,
        allowed_pages=allowed_pages,
        history=history,
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "num_predict": 300,
            "temperature": 0.3,
        },
    )

    answer = response.get("message", {}).get("content", "").strip()

    if not answer:
        return "The model returned an empty response. Please try again."

    return _remove_invalid_page_citations(
        answer,
        allowed_pages,
    )


def ask_question(question):
    """CLI mode using the saved index and global conversation history."""

    saved_index, saved_chunks = _load_saved_retrieval_system()

    answer = _generate_answer(
        question=question,
        index=saved_index,
        chunks=saved_chunks,
        history=conversation_history,
    )

    conversation_history.append({
        "role": "user",
        "content": question,
    })

    conversation_history.append({
        "role": "assistant",
        "content": answer,
    })

    return answer


def ask_uploaded_document(question, index, chunks, history):
    """
    Answer questions about a document uploaded in PaperPilot.

    Expected interface:
    ask_uploaded_document(question, index, chunks, history)
    """

    answer = _generate_answer(
        question=question,
        index=index,
        chunks=chunks,
        history=history,
    )

    if history is not None:
        history.append({
            "role": "user",
            "content": question,
        })

        history.append({
            "role": "assistant",
            "content": answer,
        })

    return answer


if __name__ == "__main__":
    print("PaperPilot Research Paper Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            answer = ask_question(question)

            print("\nAssistant:\n")
            print(answer)
            print()

        except Exception as exc:
            print(f"\nError: {exc}\n")
