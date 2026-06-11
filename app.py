"""
Streamlit UI for pdf-intelligence.

Run: streamlit run app.py

Core capability you asked for:
- Upload as many PDFs as you want from the compact bar above the chat input.
- Stay in ONE single chat conversation.
- Before every message, choose which PDF that specific message should be answered from.
- Message 1 can target "report-2023.pdf", message 2 can target "contract-v2.pdf", message 3 can go back to report-2023.pdf, etc.
- The system loads only the index of the chosen PDF for that turn.
- Chat history remembers which PDF each of your messages was targeting.
"""

import os
import shutil

import streamlit as st
from dotenv import load_dotenv

import db
from rag_core import (
    AVAILABLE_LLM_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    SAMPLE_QUESTIONS,
    build_rag_chain,
    doc_index_path,
    doc_is_indexed,
    get_embeddings,
    get_index_stats,
    get_llm,
    make_doc_id,
    pdf_file_path,
    process_pdf_to_index,
    read_pdf_bytes,
    save_pdf_bytes,
    validate_pdf_bytes,
)

load_dotenv()
db.init_db()

st.set_page_config(
    page_title="PDF Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}


def inject_styles():
    st.markdown(
        """
        <style>
            #MainMenu, footer, .stDeployButton {visibility: hidden;}

            .stApp { background-color: #212121; }

            section[data-testid="stSidebar"] {
                background-color: #171717 !important;
                border-right: 1px solid #2f2f2f;
            }
            [data-testid="stSidebarHeader"] {
                padding: 0.35rem 0.5rem 0.15rem !important;
            }
            section[data-testid="stSidebar"] .block-container {
                padding: 0.2rem 0.7rem 1rem !important;
            }
            section[data-testid="stSidebar"] h3 {
                font-size: 1.05rem;
                margin: 0 0 0.5rem;
                color: #ececec;
            }
            section[data-testid="stSidebar"] button[kind="primary"] {
                background-color: #2563eb !important;
                border-color: #2563eb !important;
                color: #ffffff !important;
            }
            section[data-testid="stSidebar"] button[kind="primary"]:hover {
                background-color: #1d4ed8 !important;
                border-color: #1d4ed8 !important;
            }

            .main .block-container { max-width: 100%; padding-top: 1rem; padding-bottom: 2rem; }

            .chat-container { max-width: 48rem; margin: 0 auto; }

            .welcome-title {
                text-align: center; font-size: 1.6rem; font-weight: 500; color: #ececec;
                margin: 3rem 0 0.4rem;
            }
            .welcome-sub {
                text-align: center; color: #8e8e8e; font-size: 0.9rem; margin-bottom: 1.25rem;
            }

            .target-bar {
                max-width: 48rem;
                margin: 0.4rem auto 0.25rem;
            }
            .target-bar [data-testid="column"] { padding: 0 0.15rem; }
            .target-bar div[data-testid="stSelectbox"] > div {
                min-height: 2.1rem;
            }
            .target-bar div[data-testid="stSelectbox"] [data-baseweb="select"] {
                font-size: 0.82rem;
            }
            .target-bar div[data-testid="stFileUploader"] {
                padding-top: 0.15rem;
            }
            .target-bar div[data-testid="stFileUploader"] section {
                padding: 0.2rem 0.35rem;
                min-height: 0;
            }
            .target-bar div[data-testid="stFileUploader"] section > div {
                font-size: 0.75rem;
            }
            .target-bar div[data-testid="stFileUploader"] button {
                font-size: 0.75rem;
                padding: 0.15rem 0.45rem;
                min-height: 1.6rem;
            }

            .sample-questions {
                max-width: 34rem;
                margin: 0 auto 0.75rem;
            }
            .sample-questions .stButton > button {
                width: 100%;
                font-size: 0.8rem;
                padding: 0.35rem 0.7rem;
                min-height: 0;
                height: auto;
                line-height: 1.35;
                text-align: left;
                white-space: normal;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cache_pdf(doc_id: str, pdf_bytes: bytes) -> None:
    st.session_state.pdf_cache[doc_id] = pdf_bytes


def get_pdf_for_doc(doc_id: str) -> bytes | None:
    if doc_id in st.session_state.pdf_cache:
        return st.session_state.pdf_cache[doc_id]
    data = read_pdf_bytes(doc_id)
    if data:
        cache_pdf(doc_id, data)
    return data


def start_new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None


def load_chat(conversation_id: int):
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = db.load_messages(conversation_id)


def build_index_for_doc(doc_id: str, pdf_bytes: bytes, filename: str,
                        chunk_size: int, chunk_overlap: int, force_rebuild: bool):
    save_pdf_bytes(doc_id, pdf_bytes)
    cache_pdf(doc_id, pdf_bytes)

    with st.spinner(f"Indexing {filename}..."):
        embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
        vectorstore = process_pdf_to_index(
            pdf_bytes,
            filename=filename,
            embeddings=embeddings,
            index_dir=doc_index_path(doc_id),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_rebuild=force_rebuild,
        )

    stats = get_index_stats(vectorstore)
    db.upsert_document(doc_id, filename, stats["chunk_count"])
    st.session_state.vectorstore = vectorstore
    return vectorstore


def index_new_upload(pdf_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int):
    if not validate_pdf_bytes(pdf_bytes):
        st.error("Uploaded file does not appear to be a valid PDF.")
        return None, None

    doc_id = make_doc_id(pdf_bytes, filename)
    cache_pdf(doc_id, pdf_bytes)

    if doc_is_indexed(doc_id):
        vs = load_vectorstore_for_doc(doc_id)
        return doc_id, vs

    vs = build_index_for_doc(doc_id, pdf_bytes, filename, chunk_size, chunk_overlap, force_rebuild=True)
    return doc_id, vs


def reindex_doc(doc_id: str, chunk_size: int, chunk_overlap: int):
    doc = db.get_document(doc_id)
    if not doc:
        return None, None
    pdf_bytes = get_pdf_for_doc(doc_id)
    if not pdf_bytes:
        st.error("Original PDF file is missing. Please upload it again.")
        return None, None

    vs = build_index_for_doc(doc_id, pdf_bytes, doc["filename"], chunk_size, chunk_overlap, force_rebuild=True)
    return doc_id, vs


def load_vectorstore_for_doc(doc_id: str):
    if "vectorstore" in st.session_state and st.session_state.get("last_loaded_doc_id") == doc_id:
        return st.session_state.vectorstore

    if not doc_is_indexed(doc_id):
        return None

    with st.spinner("Loading vector index..."):
        embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
        vectorstore = process_pdf_to_index(
            b"",
            embeddings=embeddings,
            index_dir=doc_index_path(doc_id),
            force_rebuild=False,
        )
    st.session_state.vectorstore = vectorstore
    st.session_state.last_loaded_doc_id = doc_id
    return vectorstore


def delete_document(doc_id: str):
    if os.path.exists(doc_index_path(doc_id)):
        shutil.rmtree(doc_index_path(doc_id))
    if os.path.exists(pdf_file_path(doc_id)):
        os.remove(pdf_file_path(doc_id))
    st.session_state.pdf_cache.pop(doc_id, None)
    db.delete_document(doc_id)
    if st.session_state.get("selected_doc_id") == doc_id:
        st.session_state.selected_doc_id = None
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("last_loaded_doc_id", None)
    if st.session_state.get("processed_upload_id") == doc_id:
        st.session_state.pop("processed_upload_id", None)


def render_sidebar():
    with st.sidebar:
        st.markdown("### PDF Intelligence")

        if st.button("New chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.markdown("**Recent chats**")
        for conv in db.list_conversations(limit=15):
            label = _short_label(db.get_display_title(conv["id"], conv["title"]))
            is_active = st.session_state.conversation_id == conv["id"]
            if st.button(
                label,
                key=f"chat_{conv['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                load_chat(conv["id"])
                st.rerun()

        st.markdown("---")

        llm_model = DEFAULT_LLM_MODEL
        temperature = 0.0
        top_k = 6
        chunk_size = 1000
        chunk_overlap = 150

        with st.expander("Settings", expanded=False):
            llm_model = st.selectbox("Model", AVAILABLE_LLM_MODELS, index=0)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
            top_k = st.slider("Chunks (k)", 3, 12, 6, 1)
            chunk_size = st.number_input("Chunk size", 500, 2000, 1000, 100)
            chunk_overlap = st.number_input("Overlap", 0, 400, 150, 50)

            if st.session_state.get("selected_doc_id") and st.button("Re-index selected PDF", use_container_width=True):
                st.session_state.pending_reindex = {
                    "doc_id": st.session_state.selected_doc_id,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                st.rerun()

            if st.session_state.get("selected_doc_id") and st.button("Delete selected PDF", use_container_width=True):
                delete_document(st.session_state.selected_doc_id)
                start_new_chat()
                st.rerun()

            if st.session_state.conversation_id and st.button("Delete this chat", use_container_width=True):
                db.delete_conversation(st.session_state.conversation_id)
                start_new_chat()
                st.rerun()

        return llm_model, temperature, top_k, chunk_size, chunk_overlap


def get_available_documents():
    return db.list_documents()


def _short_filename(name: str, max_len: int = 42) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _short_label(text: str, max_len: int = 34) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _chat_title_from_prompt(prompt: str, max_len: int = 48) -> str:
    return _short_label(prompt, max_len)


def _default_target_index(ids: list[str]) -> int:
    last_used = None
    for m in reversed(st.session_state.messages):
        if m.get("pdf_id") in ids:
            last_used = m["pdf_id"]
            break
    if last_used in ids:
        return ids.index(last_used)
    if st.session_state.get("selected_doc_id") in ids:
        return ids.index(st.session_state.selected_doc_id)
    return 0


def render_input_bar(available_docs: list[dict]) -> tuple[str | None, str | None, object | None]:
    """Compact PDF picker + upload row above the chat input."""
    uploaded = None
    target_doc_id = None
    target_pdf_name = None

    st.markdown('<div class="target-bar">', unsafe_allow_html=True)

    if available_docs:
        labels = {d["id"]: d["filename"] for d in available_docs}
        ids = list(labels.keys())
        default_idx = _default_target_index(ids)

        select_col, upload_col = st.columns([4.5, 1.25], gap="small")
        with select_col:
            target_doc_id = st.selectbox(
                "PDF",
                options=ids,
                index=default_idx,
                format_func=lambda x: _short_filename(labels[x]),
                key="per_turn_pdf",
                label_visibility="collapsed",
            )
            target_pdf_name = labels[target_doc_id]
            st.session_state.selected_doc_id = target_doc_id
        with upload_col:
            uploaded = st.file_uploader(
                "Add PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="pdf_upload",
            )
    else:
        upload_col, _ = st.columns([1.4, 3.6], gap="small")
        with upload_col:
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="pdf_upload",
            )

    st.markdown("</div>", unsafe_allow_html=True)
    return target_doc_id, target_pdf_name, uploaded


def render_empty_state():
    st.markdown('<p class="welcome-title">What would you like to know?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="welcome-sub">Upload a PDF below, then pick which document each message should use.</p>',
        unsafe_allow_html=True,
    )


def render_sample_questions(default_target_id):
    st.markdown('<div class="sample-questions">', unsafe_allow_html=True)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"sample_{i}"):
            st.session_state.pending_question = q
            st.session_state.pending_target_doc_id = default_target_id
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _source_page(doc) -> str:
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        return doc.metadata.get("page", "?")
    if isinstance(doc, dict):
        meta = doc.get("metadata")
        if isinstance(meta, dict):
            return meta.get("page", "?")
    return "?"


def _source_content(doc) -> str:
    if hasattr(doc, "page_content"):
        return doc.page_content
    if isinstance(doc, dict):
        return doc.get("page_content", "")
    return ""


def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("pdf_name"):
            st.caption(f"→ {msg['pdf_name']}")
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i} · page {_source_page(doc)}**")
                    preview = _source_content(doc)
                    if len(preview) > 480:
                        preview = preview[:480] + "..."
                    st.caption(preview)


def ensure_conversation(pdf_id: str | None, pdf_name: str | None) -> int:
    if st.session_state.conversation_id is not None:
        return st.session_state.conversation_id
    title = "New chat"
    cid = db.create_conversation(title, pdf_id=pdf_id, pdf_name=pdf_name)
    st.session_state.conversation_id = cid
    return cid


def submit_user_message(prompt: str, target_doc_id: str, target_pdf_name: str) -> None:
    conv_id = ensure_conversation(target_doc_id, target_pdf_name)
    conv = db.get_conversation(conv_id)
    if conv and conv["title"] == "New chat":
        if not st.session_state.messages:
            db.update_conversation_title(conv_id, _chat_title_from_prompt(prompt))
        else:
            first_user = next(
                (m["content"] for m in st.session_state.messages if m["role"] == "user"),
                prompt,
            )
            db.update_conversation_title(conv_id, _chat_title_from_prompt(first_user))
    db.add_message(conv_id, "user", prompt, pdf_id=target_doc_id, pdf_name=target_pdf_name)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "pdf_id": target_doc_id,
        "pdf_name": target_pdf_name,
    })


def run_pending_generation() -> None:
    """Generate the assistant reply above the input, then rerun with full history."""
    gen = st.session_state.pop("generating")
    vs = load_vectorstore_for_doc(gen["target_doc_id"])
    if vs is None:
        st.error("Could not load the vector index for the selected PDF.")
        return

    rag = build_rag_chain(
        vs,
        get_llm(model=gen["llm_model"], temperature=gen["temperature"]),
        k=gen["top_k"],
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.invoke({"input": gen["prompt"]})
            answer = result["answer"].strip()
            sources = result.get("context", [])

    conv_id = st.session_state.conversation_id
    db.add_message(
        conv_id,
        "assistant",
        answer,
        pdf_id=gen["target_doc_id"],
        pdf_name=gen["target_pdf_name"],
        sources=sources,
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pdf_id": gen["target_doc_id"],
        "pdf_name": gen["target_pdf_name"],
        "sources": sources,
    })
    st.rerun()


def main():
    inject_styles()

    llm_model, temperature, top_k, chunk_size, chunk_overlap = render_sidebar()

    pending_reindex = st.session_state.pop("pending_reindex", None)
    if pending_reindex:
        doc_id, vs = reindex_doc(
            pending_reindex["doc_id"],
            pending_reindex["chunk_size"],
            pending_reindex["chunk_overlap"],
        )
        if vs:
            st.session_state.selected_doc_id = doc_id
            st.session_state.last_loaded_doc_id = doc_id
            st.toast("PDF re-indexed.")
        st.rerun()

    available_docs = get_available_documents()

    # Chat area
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            render_message(msg)

        if st.session_state.get("generating"):
            run_pending_generation()

        has_messages = bool(st.session_state.messages)
        if not has_messages and not st.session_state.get("generating"):
            render_empty_state()

        if not has_messages and available_docs:
            default_id = st.session_state.get("selected_doc_id") or available_docs[0]["id"]
            render_sample_questions(default_id)

        target_doc_id, target_pdf_name, uploaded = render_input_bar(available_docs)

        if uploaded is not None:
            pdf_bytes = uploaded.getvalue()
            upload_id = make_doc_id(pdf_bytes, uploaded.name)
            if st.session_state.get("processed_upload_id") != upload_id:
                doc_id, _ = index_new_upload(pdf_bytes, uploaded.name, chunk_size, chunk_overlap)
                if doc_id:
                    st.session_state.selected_doc_id = doc_id
                    st.session_state.processed_upload_id = upload_id
                    st.rerun()

        prompt = st.session_state.pop("pending_question", None)
        pending_target = st.session_state.pop("pending_target_doc_id", None)
        if pending_target and available_docs:
            target_doc_id = pending_target
            target_pdf_name = next((d["filename"] for d in available_docs if d["id"] == target_doc_id), None)

        if prompt is None:
            ph = "Ask anything" if available_docs else "Upload a PDF to get started"
            prompt = st.chat_input(ph, disabled=not available_docs)

        if prompt and available_docs and target_doc_id:
            if not doc_is_indexed(target_doc_id):
                st.error("Index for the selected PDF could not be loaded. Try re-indexing it from Settings.")
            else:
                submit_user_message(prompt, target_doc_id, target_pdf_name)
                st.session_state.generating = {
                    "prompt": prompt,
                    "target_doc_id": target_doc_id,
                    "target_pdf_name": target_pdf_name,
                    "llm_model": llm_model,
                    "temperature": temperature,
                    "top_k": top_k,
                }
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
