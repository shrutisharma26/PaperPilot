PaperPilot — Research Reimagined
PaperPilot is a research-paper assistant that lets users upload PDF papers, ask questions about them, and explore their contents using Retrieval-Augmented Generation (RAG).

Instead of repeatedly searching through a long paper, users can ask natural-language questions and receive answers grounded in retrieved passages, with page citations where available.
Features
- PDF upload and processing — upload research papers for text extraction and indexing.
- Semantic search — find relevant paper passages using embeddings and FAISS.
- Grounded Q&A — ask questions about a paper using a local LLM.
- Page citations — answers can reference source pages using citations such as [Page 4].
- Paper library — keep multiple PDFs available during the current app session and switch between them.
- Per-paper chat history — conversations are maintained separately for each paper during the session.
- Cross-paper questions — compare or ask questions across papers in the library.
- Summary toolkit — generate a TL;DR, methodology overview, key findings, contributions, and limitations.
- Export — download the conversation or assistant answers/summaries as Markdown.
- Streamlit interface — interactive web app with a custom dark-themed UI.
Current limitation: the document library and conversations are session-based. They are not persistent across all app/browser restarts unless persistence is added. Citation quality depends on the retrieved passages and PDF page metadata.

Tech stack
- Python
- Streamlit
- PyMuPDF (pymupdf) for PDF text extraction
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- FAISS for vector similarity search
- Ollama for local LLM inference
- qwen2.5:3b as the chat model