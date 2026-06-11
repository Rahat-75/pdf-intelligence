"""
RAG pipeline: PDF → chunks → embeddings → FAISS → retrieve → Gemini.
"""

import hashlib
import os
import tempfile
import time
from typing import Optional, Union

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# --- Configuration ---
INDEX_DIR = "faiss_index"
PDF_STORE_DIR = "pdfs"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

AVAILABLE_LLM_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemma-4-31b-it",
]
DEFAULT_LLM_MODEL = AVAILABLE_LLM_MODELS[0]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 6

SYSTEM_PROMPT = (
    "You are a document Q&A assistant.\n"
    "Answer using only the retrieved context below.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Be concise, factual, and cite relevant phrases when helpful.\n\n"
    "Context:\n{context}"
)

SAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key points in three bullet points.",
    "What conclusions or recommendations are mentioned?",
]


def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
    """Return a Gemini embedding client (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY applied per call)."""
    return GoogleGenerativeAIEmbeddings(model=model)


def get_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model for answer generation."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _load_documents_from_path(pdf_path: str) -> list[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(pdf_path).load()


def _load_documents_from_bytes(data: bytes, filename: str = "uploaded.pdf") -> list[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    try:
        return _load_documents_from_path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def validate_pdf_bytes(data: bytes) -> bool:
    if not data or len(data) < 100 or not data.startswith(b"%PDF"):
        return False
    try:
        import io
        from pypdf import PdfReader

        PdfReader(io.BytesIO(data))
        return True
    except Exception:
        return False


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def create_faiss_index(
    documents: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    return FAISS.from_documents(documents, embeddings)


def save_faiss_index(vectorstore: FAISS, path: str = INDEX_DIR) -> None:
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_faiss_index(
    embeddings: GoogleGenerativeAIEmbeddings,
    path: str = INDEX_DIR,
) -> FAISS:
    if not index_exists(path):
        raise FileNotFoundError(f"No FAISS index found at '{path}'")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def doc_index_path(doc_id: str) -> str:
    return os.path.join(INDEX_DIR, doc_id)


def pdf_file_path(doc_id: str) -> str:
    return os.path.join(PDF_STORE_DIR, f"{doc_id}.pdf")


def make_doc_id(pdf_bytes: bytes, filename: str) -> str:
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    return f"{safe[:40]}_{digest}"


def save_pdf_bytes(doc_id: str, pdf_bytes: bytes) -> str:
    if not validate_pdf_bytes(pdf_bytes):
        raise ValueError("Invalid or corrupted PDF data.")
    os.makedirs(PDF_STORE_DIR, exist_ok=True)
    path = pdf_file_path(doc_id)

    if os.path.exists(path):
        try:
            with open(path, "rb") as existing:
                if existing.read() == pdf_bytes:
                    return path
        except OSError:
            pass

    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=PDF_STORE_DIR, prefix=f".{doc_id}_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pdf_bytes)
            f.flush()
            os.fsync(f.fileno())

        last_error: OSError | None = None
        for attempt in range(6):
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.replace(tmp_path, path)
                tmp_path = ""
                return path
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.15 * (attempt + 1))
        if last_error:
            raise last_error
        raise PermissionError(f"Could not write PDF to '{path}'")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def read_pdf_bytes(doc_id: str) -> bytes | None:
    path = pdf_file_path(doc_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    if not validate_pdf_bytes(data):
        return None
    return data


def index_exists(path: str = INDEX_DIR) -> bool:
    return os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss"))


def doc_is_indexed(doc_id: str) -> bool:
    return index_exists(doc_index_path(doc_id))


def get_index_stats(vectorstore: FAISS) -> dict:
    return {"chunk_count": vectorstore.index.ntotal}


def process_pdf_to_index(
    source: Union[str, bytes],
    *,
    filename: Optional[str] = None,
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
    index_dir: str = INDEX_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Build or load a FAISS vector store from a PDF path or bytes.

    When an index already exists and force_rebuild is False, loads from disk.
    Otherwise parses the PDF, chunks text, embeds with Gemini, and persists locally.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if not force_rebuild and index_exists(index_dir):
        return load_faiss_index(embeddings, index_dir)

    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        if not validate_pdf_bytes(raw):
            raise ValueError("Invalid or corrupted PDF file.")
        docs = _load_documents_from_bytes(raw, filename or "uploaded.pdf")
    else:
        path = str(source)
        with open(path, "rb") as f:
            raw = f.read()
        if not validate_pdf_bytes(raw):
            raise ValueError(f"Invalid or corrupted PDF on disk: {path}")
        docs = _load_documents_from_bytes(raw, filename or os.path.basename(path))

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = create_faiss_index(chunks, embeddings)
    save_faiss_index(vectorstore, index_dir)
    return vectorstore


def build_rag_chain(
    vectorstore: FAISS,
    llm: ChatGoogleGenerativeAI,
    k: int = DEFAULT_TOP_K,
):
    """Build a retrieval chain: retrieve top-k chunks → stuff into prompt → generate answer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, combine_docs_chain)


def ask_question(
    vectorstore: FAISS,
    llm: ChatGoogleGenerativeAI,
    question: str,
    k: int = DEFAULT_TOP_K,
) -> dict:
    """Ask a question and return the chain result (`answer` + retrieved `context`)."""
    chain = build_rag_chain(vectorstore, llm, k=k)
    return chain.invoke({"input": question})
