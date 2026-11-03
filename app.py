import streamlit as st
import html
import ollama

from document_manager import process_pdf
from rag_pipeline import ask_uploaded_document


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PaperPilot — Research Reimagined",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "document_index": None,
    "document_chunks": None,
    "document_name": None,
    "document_id": None,
    "messages": [],
    "rag_history": [],
    "pending_question": None,
    "document_library": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def save_active_document_history():
    """Persist the active paper's chat and RAG history in its library entry."""
    active_id = st.session_state.get("document_id")
    if active_id is None:
        return

    for document in st.session_state.get("document_library", []):
        if document.get("id") == active_id:
            document["messages"] = list(st.session_state.get("messages", []))
            document["rag_history"] = list(st.session_state.get("rag_history", []))
            break


def load_document_history(document):
    """Restore saved chat history when opening a paper."""
    st.session_state.messages = list(document.get("messages", []))
    st.session_state.rag_history = list(document.get("rag_history", []))
    st.session_state.pending_question = None


# Backfill history fields for papers added before per-document chat history.
for _document in st.session_state.document_library:
    _document.setdefault("messages", [])
    _document.setdefault("rag_history", [])


# --------------------------------------------------
# CUSTOM CSS
# Palette inspired by the reference schemes:
#   plum-black bg, rose #DA7B93, coral #F58F7C, orchid #D174D2,
#   teal #376E6F, indigo #3F567F, violet #412653, ember #E0563F
# --------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #15111A;
    --glass: rgba(255, 255, 255, 0.045);
    --glass-strong: rgba(255, 255, 255, 0.075);
    --glass-border: rgba(255, 255, 255, 0.10);
    --white: #F7F2F4;
    --muted: #B3A9B8;
    --rose: #DA7B93;
    --coral: #F58F7C;
    --orchid: #D174D2;
    --teal: #376E6F;
    --indigo: #3F567F;
    --violet: #412653;
    --ember: #E0563F;
    --grad: linear-gradient(120deg, #F58F7C 0%, #DA7B93 45%, #D174D2 100%);
}

/* ---------- base ---------- */
.stApp {
    background: var(--bg);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
    overflow-x: hidden;
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], .stDeployButton { display: none !important; }

.block-container {
    max-width: 1180px;
    padding: 1.5rem 2.2rem 4rem;
    position: relative;
    z-index: 2;
}

/* ---------- animated gradient blobs behind everything ---------- */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: .55;
    z-index: 0;
    pointer-events: none;
}
.stApp::before {
    width: 620px; height: 620px;
    top: -180px; right: -140px;
    background: radial-gradient(circle at 30% 30%, #DA7B93, #412653 70%);
    animation: drift1 18s ease-in-out infinite alternate;
}
.stApp::after {
    width: 560px; height: 560px;
    bottom: -200px; left: -160px;
    background: radial-gradient(circle at 60% 40%, #376E6F, #1C3334 70%);
    animation: drift2 22s ease-in-out infinite alternate;
}
@keyframes drift1 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(-90px, 120px) scale(1.18); }
}
@keyframes drift2 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(120px, -90px) scale(1.12); }
}

/* ---------- navbar ---------- */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 10px 0 22px;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 48px;
}
.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -1px;
}
.brand span {
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.nav-label {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 2.5px;
    text-align: right;
}

/* ---------- hero ---------- */
.eyebrow {
    font-size: 12px;
    letter-spacing: 4px;
    color: var(--coral);
    font-weight: 700;
    margin-bottom: 22px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(48px, 7vw, 88px);
    font-weight: 700;
    line-height: .98;
    letter-spacing: -4px;
    margin-bottom: 24px;
    color: var(--white);
    text-transform: uppercase;
}
.hero-title .accent {
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-description {
    color: var(--muted);
    font-size: 17px;
    line-height: 1.75;
    max-width: 560px;
    margin-bottom: 26px;
}
.tag {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 30px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(8px);
    color: #E9D9E6;
    font-size: 12px;
    margin: 4px 6px 4px 0;
}

/* ---------- hero visual: blob card + swatch strip ---------- */
.hero-visual {
    position: relative;
    display: flex;
    height: 430px;
    border-radius: 28px;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    box-shadow: 0 30px 80px rgba(0, 0, 0, .45);
}
.blob-stage {
    position: relative;
    flex: 1;
    background: linear-gradient(160deg, #2F4454, #2E151B);
    overflow: hidden;
}
.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(2px);
}
.b1 { width: 230px; height: 230px; top: -50px; left: -30px;
      background: radial-gradient(circle at 35% 30%, #DA7B93, #5a2236 75%);
      animation: float1 9s ease-in-out infinite alternate; }
.b2 { width: 200px; height: 200px; top: 90px; left: 110px;
      background: radial-gradient(circle at 40% 30%, #4aa0a1, #1C3334 75%);
      animation: float2 11s ease-in-out infinite alternate; }
.b3 { width: 170px; height: 170px; bottom: -30px; left: 20px;
      background: radial-gradient(circle at 40% 30%, #D174D2, #412653 75%);
      animation: float2 10s ease-in-out infinite alternate-reverse; }
.b4 { width: 120px; height: 120px; bottom: 70px; right: 20px;
      background: radial-gradient(circle at 40% 30%, #F58F7C, #E0563F 80%);
      animation: float1 12s ease-in-out infinite alternate-reverse; }
@keyframes float1 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(24px, 30px) scale(1.08); }
}
@keyframes float2 {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(-28px, -22px) scale(1.1); }
}
.stage-caption {
    position: absolute;
    left: 24px; bottom: 22px; right: 24px;
    z-index: 3;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 26px;
    line-height: 1.1;
    text-transform: uppercase;
    text-shadow: 0 4px 24px rgba(0, 0, 0, .6);
}
.stage-caption small {
    display: block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    letter-spacing: 3px;
    font-weight: 500;
    color: #E6D6E0;
    margin-top: 8px;
}

/* ---------- feature cards ---------- */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 34px 0 8px;
}
.feature-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 22px;
    backdrop-filter: blur(10px);
    transition: transform .25s ease, border-color .25s ease, background .25s ease;
}
.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(218, 123, 147, .55);
    background: var(--glass-strong);
}
.feature-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin-bottom: 14px;
    color: #15111A;
}
.fc1 .feature-icon { background: #F58F7C; }
.fc2 .feature-icon { background: #D174D2; }
.fc3 .feature-icon { background: #4fb3b3; }
.feature-card h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    margin: 0 0 6px;
    color: var(--white);
}
.feature-card p {
    color: var(--muted);
    font-size: 13.5px;
    line-height: 1.6;
    margin: 0;
}

/* ---------- upload ---------- */
.upload-panel {
    background: linear-gradient(135deg, rgba(65, 38, 83, .75), rgba(46, 21, 27, .75));
    border: 1px solid rgba(209, 116, 210, .35);
    border-radius: 24px;
    padding: 30px;
    margin-top: 40px;
    backdrop-filter: blur(12px);
}
.upload-symbol {
    width: 46px; height: 46px;
    border-radius: 14px;
    background: var(--grad);
    color: #15111A;
    font-size: 24px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 14px;
}
.upload-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(22px, 3vw, 30px);
    font-weight: 700;
    margin-bottom: 8px;
}
.upload-subtitle {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
}

[data-testid="stFileUploader"] {
    background: rgba(218, 123, 147, 0.07);
    border: 1.5px dashed rgba(245, 143, 124, .6);
    border-radius: 18px;
    padding: 16px;
    margin-top: 14px;
    transition: .25s ease;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(218, 123, 147, 0.13);
    border-color: var(--coral);
}
[data-testid="stFileUploader"] section { background: transparent; }
[data-testid="stFileUploader"] button {
    background: var(--grad) !important;
    color: #15111A !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* ---------- steps ---------- */
.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 34px;
}
.step {
    border-top: 1px solid var(--glass-border);
    padding-top: 14px;
}
.step b {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.step span { display: block; color: var(--muted); font-size: 13px; margin-top: 4px; }

/* ---------- buttons ---------- */
.stButton button {
    border-radius: 12px;
    border: 1px solid rgba(245, 143, 124, .45);
    background: var(--glass);
    color: var(--white);
    font-weight: 600;
    min-height: 44px;
    backdrop-filter: blur(8px);
    transition: .22s ease;
}
.stButton button:hover {
    background: var(--grad);
    color: #15111A;
    border-color: transparent;
    transform: translateY(-2px);
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}
[data-testid="stChatInput"] {
    border: 1px solid rgba(209, 116, 210, .45);
    border-radius: 16px;
    background: rgba(21, 17, 26, .85);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--coral);
    box-shadow: 0 0 0 3px rgba(245, 143, 124, .15);
}

/* ---------- workspace ---------- */
.document-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(135deg, rgba(55, 110, 111, .35), rgba(65, 38, 83, .45));
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 20px 22px;
    margin: 20px 0;
    overflow-wrap: anywhere;
}
.doc-icon {
    width: 48px; height: 48px;
    flex-shrink: 0;
    border-radius: 14px;
    background: var(--grad);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.doc-label { color: var(--coral); font-size: 11px; letter-spacing: 3px; font-weight: 700; }
.doc-name { font-size: 18px; font-weight: 600; margin-top: 6px; }
.doc-meta { color: var(--muted); font-size: 13px; margin-top: 4px; }

.try-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 600;
    margin: 6px 0 12px;
}

hr { border-color: var(--glass-border) !important; }

.footer-note {
    text-align: center;
    color: #857A8B;
    font-size: 11px;
    letter-spacing: 3px;
    margin-top: 44px;
    line-height: 1.8;
}

/* ---------- tablet ---------- */
@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1.5rem 3rem; }
    .navbar { margin-bottom: 34px; }
    .hero-title { letter-spacing: -3px; }
    .hero-visual { height: 340px; margin-top: 28px; }
    .feature-grid, .steps { grid-template-columns: 1fr; }
}

/* ---------- mobile ---------- */
@media (max-width: 600px) {
    .block-container { padding: .8rem 1rem 2.5rem; }
    .navbar { padding-bottom: 16px; margin-bottom: 26px; align-items: flex-start; }
    .brand { font-size: 22px; }
    .nav-label { font-size: 9px; letter-spacing: 1px; max-width: 130px; }
    .eyebrow { font-size: 10px; letter-spacing: 2.5px; margin-bottom: 16px; }
    .hero-title { font-size: clamp(40px, 12vw, 56px); letter-spacing: -2.5px; }
    .hero-description { font-size: 15px; line-height: 1.6; }
    .tag { font-size: 11px; padding: 7px 10px; }
    .hero-visual { height: 280px; }
    .stage-caption { font-size: 20px; }
    .upload-panel { padding: 20px; border-radius: 18px; margin-top: 26px; }
    .document-card { padding: 14px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

st.markdown(
    """<div class="navbar"><div class="brand">Paper<span>Pilot</span> ✳</div><div class="nav-label">RESEARCH INTELLIGENCE · AI</div></div>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LANDING / PDF UPLOAD
# --------------------------------------------------

if st.session_state.document_index is None:

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            """<div class="eyebrow">✳ YOUR RESEARCH, REIMAGINED</div><div class="hero-title">Think<br>beyond<br><span class="accent">the paper.</span></div><div class="hero-description">Research papers are complicated. Understanding them shouldn't be. Upload a paper and explore its ideas with your AI research companion.</div><div><span class="tag">✦ AI-powered insights</span><span class="tag">↗ Page citations</span><span class="tag">◈ Research assistant</span></div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """<div class="hero-visual"><div class="blob-stage"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="blob b4"></div><div class="stage-caption">Let's be<br>curious<small>PAPERPILOT / 001</small></div></div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """<div class="feature-grid"><div class="feature-card fc1"><div class="feature-icon">✦</div><h4>Instant summaries</h4><p>Get the core idea, method and results of any paper in seconds.</p></div><div class="feature-card fc2"><div class="feature-icon">↗</div><h4>Page-level citations</h4><p>Every answer points back to where it came from in the PDF.</p></div><div class="feature-card fc3"><div class="feature-icon">◈</div><h4>Ask anything</h4><p>Follow-up questions that remember the conversation so far.</p></div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="upload-panel"><div class="upload-symbol">↥</div><div class="upload-title">Bring your research to life.</div><div class="upload-subtitle">Upload a PDF to start exploring, questioning, and discovering. Your document is processed locally by this app.</div></div>""",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop your research papers here or browse files",
        type=["pdf"],
        accept_multiple_files=True,
        key="main_pdf_upload",
        help="Choose one or more PDF research papers.",
    )

    if uploaded_files:
        papers_added = 0

        for uploaded_file in uploaded_files:
            document_id = (uploaded_file.name, uploaded_file.size)

            # If this paper is already in the library, reuse its saved index.
            existing_doc = next(
                (doc for doc in st.session_state.document_library if doc["id"] == document_id),
                None
            )

            if existing_doc:
                st.session_state.document_index = existing_doc["index"]
                st.session_state.document_chunks = existing_doc["chunks"]
                st.session_state.document_name = existing_doc["name"]
                st.session_state.document_id = existing_doc["id"]
                continue

            try:
                with st.spinner(f"Processing {uploaded_file.name}…"):
                    index, chunks = process_pdf(uploaded_file.getvalue())

                new_doc = {
                    "id": document_id,
                    "name": uploaded_file.name,
                    "index": index,
                    "chunks": chunks,
                    "messages": [],
                    "rag_history": [],
                }
                st.session_state.document_library.append(new_doc)

                # Save current paper history before switching to the new paper.
                save_active_document_history()

                # Make the newly processed paper the active document.
                st.session_state.document_index = index
                st.session_state.document_chunks = chunks
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_id = document_id
                papers_added += 1

            except Exception as error:
                st.error(f"Could not process {uploaded_file.name}: {error}")

        st.session_state.messages = []
        st.session_state.rag_history = []
        st.session_state.pending_question = None

        if papers_added:
            st.success(f"Added {papers_added} paper(s) to your library!")
            st.rerun()

    # Reopen papers already processed during this session
    if st.session_state.document_library:
        st.divider()
        st.markdown('<div class="try-label">Continue with a paper from your library</div>', unsafe_allow_html=True)

        for i, doc in enumerate(st.session_state.document_library):
            safe_name = html.escape(str(doc.get("name", "Untitled paper")))
            library_col, open_col, remove_col = st.columns([3, 1, 1])

            with library_col:
                st.markdown(
                    f"""
                    <div class="document-card">
                        <div class="doc-icon">📄</div>
                        <div>
                            <div class="doc-name">{safe_name}</div>
                            <div class="doc-meta">{len(doc.get("chunks", []))} searchable text segments</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with open_col:
                if st.button("Open", key=f"landing_open_doc_{i}", use_container_width=True):
                    save_active_document_history()
                    st.session_state.document_index = doc["index"]
                    st.session_state.document_chunks = doc["chunks"]
                    st.session_state.document_name = doc["name"]
                    st.session_state.document_id = doc["id"]
                    load_document_history(doc)
                    st.rerun()

            with remove_col:
                if st.button("Remove", key=f"landing_remove_doc_{i}", use_container_width=True):
                    removed_id = doc["id"]
                    st.session_state.document_library = [
                        item for item in st.session_state.document_library
                        if item["id"] != removed_id
                    ]
                    if st.session_state.document_id == removed_id:
                        st.session_state.document_index = None
                        st.session_state.document_chunks = None
                        st.session_state.document_name = None
                        st.session_state.document_id = None
                        st.session_state.messages = []
                        st.session_state.rag_history = []
                        st.session_state.pending_question = None
                    st.rerun()

    st.markdown(
        """<div class="steps"><div class="step"><b>01</b><span>Upload your PDF</span></div><div class="step"><b>02</b><span>We index it into searchable passages</span></div><div class="step"><b>03</b><span>Ask questions, get cited answers</span></div></div><div class="footer-note">UNDERSTAND MORE · READ SMARTER · DISCOVER FASTER</div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# RESEARCH CHAT
# --------------------------------------------------

else:

    st.markdown(
        """<div class="eyebrow">✳ YOUR RESEARCH WORKSPACE</div><div class="hero-title" style="font-size:clamp(38px,5vw,60px);">Let's explore <span class="accent">your paper.</span></div>""",
        unsafe_allow_html=True,
    )

    safe_document_name = html.escape(str(st.session_state.document_name or "Uploaded paper"))
    segment_count = len(st.session_state.document_chunks or [])

    if segment_count == 0:
        st.warning(
            "This PDF did not produce searchable text. It may be scanned/image-based. "
            "Try a text-based PDF or add OCR support."
        )

    st.markdown(
        f"""<div class="document-card"><div class="doc-icon">📄</div><div><div class="doc-label">CURRENT DOCUMENT</div><div class="doc-name">{safe_document_name}</div><div class="doc-meta">{segment_count} searchable text segments · {len(st.session_state.messages) // 2} questions asked</div></div></div>""",
        unsafe_allow_html=True,
    )

    upload_col, tip_col = st.columns([1, 2])

    with upload_col:
        if st.button("＋ Upload a different paper", use_container_width=True):
            st.session_state.document_index = None
            st.session_state.document_chunks = None
            st.session_state.document_name = None
            st.session_state.document_id = None
            st.session_state.messages = []
            st.session_state.rag_history = []
            st.session_state.pending_question = None
            st.session_state.main_pdf_upload = None
            st.rerun()

    with tip_col:
        st.caption("Ask questions, explore concepts, or request a summary.")

    st.divider()

    # Document Library
    st.markdown('<div class="try-label">Your document library</div>', unsafe_allow_html=True)

    if st.session_state.document_library:
        for i, doc in enumerate(st.session_state.document_library):
            safe_library_name = html.escape(str(doc.get("name", "Untitled paper")))
            is_current = doc["id"] == st.session_state.document_id

            st.markdown(
                f"""
                <div class="document-card">
                    <div class="doc-icon">📄</div>
                    <div>
                        <div class="doc-name">{safe_library_name}</div>
                        <div class="doc-meta">{len(doc.get("chunks", []))} searchable text segments</div>
                        <div class="doc-meta">{"Currently open" if is_current else "Saved in your library"}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            action_col, remove_col = st.columns([3, 1])

            with action_col:
                if not is_current and st.button(
                    f"Open {doc.get('name', 'paper')}",
                    key=f"open_library_doc_{i}",
                    use_container_width=True
                ):
                    save_active_document_history()
                    st.session_state.document_index = doc["index"]
                    st.session_state.document_chunks = doc["chunks"]
                    st.session_state.document_name = doc["name"]
                    st.session_state.document_id = doc["id"]
                    load_document_history(doc)
                    st.rerun()
                elif is_current:
                    st.caption("This paper is currently open.")

            with remove_col:
                if st.button("Remove", key=f"workspace_remove_doc_{i}", use_container_width=True):
                    removed_id = doc["id"]
                    st.session_state.document_library = [
                        item for item in st.session_state.document_library
                        if item["id"] != removed_id
                    ]
                    if st.session_state.document_id == removed_id:
                        st.session_state.document_index = None
                        st.session_state.document_chunks = None
                        st.session_state.document_name = None
                        st.session_state.document_id = None
                        st.session_state.messages = []
                        st.session_state.rag_history = []
                        st.session_state.pending_question = None
                    st.rerun()
    else:
        st.caption("Your uploaded papers will appear here.")

    st.divider()

    multi_paper_mode = False
    if len(st.session_state.document_library) > 1:
        multi_paper_mode = st.checkbox(
            "Compare / ask across all papers in my library",
            value=False,
            help="When enabled, PaperPilot retrieves an answer from each paper and combines them."
        )
        if multi_paper_mode:
            st.info(f"Cross-paper mode is enabled for {len(st.session_state.document_library)} papers.")

    st.markdown('<div class="try-label">PAPER SUMMARY TOOLKIT</div>', unsafe_allow_html=True)
    st.caption("Generate a focused summary of a specific part of the paper.")

    summary_actions = [
        ("⚡ TL;DR", "Give me a concise TL;DR of this paper in 5 bullet points. Include the research problem, approach, key result, and significance. Cite the relevant pages."),
        ("🧪 Methodology", "Explain the methodology step by step. Include the data, methods, models, and experimental setup if available. Cite the relevant pages."),
        ("📊 Key findings", "Summarize the paper's key findings and results. Include important quantitative results only if stated in the excerpts. Cite the relevant pages."),
        ("💡 Contributions", "What are the paper's main contributions and what is novel about its approach? Cite the relevant pages."),
        ("⚠️ Limitations", "Identify limitations, assumptions, and future work explicitly discussed by the authors. If the excerpts do not state these, say so. Cite the relevant pages."),
    ]

    summary_cols = st.columns(5)
    for col, (label, prompt_text) in zip(summary_cols, summary_actions):
        with col:
            if st.button(label, use_container_width=True, key=f"summary_tool_{label}"):
                st.session_state.pending_question = prompt_text
                st.rerun()

    if not st.session_state.messages:
        st.markdown('<div class="try-label">TRY ASKING</div>', unsafe_allow_html=True)

        suggestions = [
            "What is the research problem?",
            "What is its main contribution?",
            "Explain a key concept",
        ]

        prompt_cols = st.columns(3)

        for col, suggestion in zip(prompt_cols, suggestions):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"starter_{suggestion}"):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Export conversation and assistant-generated summaries.
    if st.session_state.messages:
        export_col1, export_col2 = st.columns(2)

        current_paper = st.session_state.get("document_name") or "PaperPilot"
        safe_export_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in current_paper
        )

        chat_lines = [f"# PaperPilot Chat — {current_paper}", ""]
        for message in st.session_state.messages:
            role = "You" if message["role"] == "user" else "PaperPilot"
            chat_lines.extend([f"## {role}", "", message["content"], ""])
        chat_markdown = "\\n".join(chat_lines)

        with export_col1:
            st.download_button(
                "⬇️ Export chat (.md)",
                data=chat_markdown,
                file_name=f"{safe_export_name}_paperpilot_chat.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_chat_markdown",
            )

        assistant_answers = [
            message["content"]
            for message in st.session_state.messages
            if message["role"] == "assistant"
        ]
        summaries_markdown = (
            f"# PaperPilot Answers & Summaries — {current_paper}\\n\\n"
            + "\\n\\n---\\n\\n".join(assistant_answers)
        )

        with export_col2:
            st.download_button(
                "📄 Export answers & summaries (.md)",
                data=summaries_markdown,
                file_name=f"{safe_export_name}_paperpilot_summaries.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_summaries_markdown",
            )

    # Render conversation so far.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about your paper…")

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        save_active_document_history()

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                if multi_paper_mode:
                    paper_answers = []
                    with st.spinner("Reading each paper and comparing the findings…"):
                        for doc in st.session_state.document_library:
                            doc_answer = ask_uploaded_document(
                                question,
                                doc["index"],
                                doc["chunks"],
                                [],
                            )
                            paper_answers.append(
                                f"### Paper: {doc['name']}\\n{doc_answer}"
                            )

                        combined_context = "\\n\\n".join(paper_answers)
                        synthesis_prompt = (
                            "You are PaperPilot, a research assistant comparing multiple papers. "
                            "Answer using only the paper-specific responses below. Compare similarities "
                            "and differences when relevant. Clearly identify which paper supports each "
                            "point. If information is missing, say so instead of guessing.\\n\\n"
                            f"Question: {question}\\n\\nPaper-specific findings:\\n{combined_context}"
                        )

                        final_response = ollama.chat(
                            model="qwen2.5:3b",
                            messages=[{"role": "user", "content": synthesis_prompt}],
                            options={"num_predict": 450, "temperature": 0.2},
                        )
                        answer = final_response["message"]["content"].strip()
                else:
                    with st.spinner("Searching the paper and generating an answer…"):
                        answer = ask_uploaded_document(
                            question,
                            st.session_state.document_index,
                            st.session_state.document_chunks,
                            st.session_state.rag_history,
                        )

                st.markdown(answer)

            except Exception as error:
                error_text = str(error).lower()
                if "connection" in error_text or "ollama" in error_text:
                    answer = (
                        "⚠️ I couldn't connect to Ollama. Please make sure Ollama is running "
                        "and that the `qwen2.5:3b` model is installed. In PowerShell, try "
                        "`ollama list` and then restart PaperPilot."
                    )
                else:
                    answer = (
                        "⚠️ Something went wrong while answering. Please try again. "
                        "If it keeps happening, check the terminal for the detailed error."
                    )
                st.warning(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_active_document_history()
