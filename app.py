import streamlit as st
import os
import asyncio
import uuid
import shutil
import time
import re
from dotenv import load_dotenv

load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 10
MAX_TOTAL_SIZE_MB = 50
MAX_FILES         = 10
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
TOP_K             = 4

PREFERRED_MODELS = [
    "gemini-2.0-flash-lite",   # highest free tier RPM — try first
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A Chat", page_icon="📄", layout="wide")
st.title("📘 Ask Your PDFs")
st.write(
    f"Upload up to **{MAX_FILES} PDFs** "
    f"(max {MAX_FILE_SIZE_MB} MB each, {MAX_TOTAL_SIZE_MB} MB total) "
    "and ask questions across all of them."
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "messages":      [],
    "retriever":     None,
    "vectorstore":   None,
    "indexed_files": [],
    "chroma_dir":    None,
    "active_model":  None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── API key check ─────────────────────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "❌ GOOGLE_API_KEY not found.\n\n"
        "Create a `.env` file in your project root:\n\n"
        "`GOOGLE_API_KEY=AIza...your_key_here`\n\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ── Model detection ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model_name():
    try:
        available = {
            m.name.replace("models/", "")
            for m in client.models.list()
        }
        for candidate in PREFERRED_MODELS:
            if candidate in available:
                return candidate
        if available:
            return next(iter(available))
    except Exception as e:
        st.error(f"❌ Could not list models: {e}")
        st.stop()
    return None

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

model_name = get_model_name()
if not model_name:
    st.error("❌ No compatible Gemini model found for your API key.")
    st.stop()

# Set active model from session or default
if st.session_state.active_model is None:
    st.session_state.active_model = model_name

# ── Safe ChromaDB cleanup (handles Windows WinError 32) ──────────────────────
def safe_delete_chroma(directory: str):
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore._client.close()
        except Exception:
            pass
        st.session_state.vectorstore = None
        st.session_state.retriever   = None

    if directory and os.path.exists(directory):
        for attempt in range(5):
            try:
                shutil.rmtree(directory)
                break
            except PermissionError:
                time.sleep(0.5)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Everything"):
        safe_delete_chroma(st.session_state.chroma_dir)
        st.session_state.messages      = []
        st.session_state.indexed_files = []
        st.session_state.chroma_dir    = None
        st.session_state.active_model  = model_name
        st.rerun()

    st.markdown("---")
    st.markdown(f"**🤖 Active model:** `{st.session_state.active_model}`")
    st.markdown("---")

    if st.session_state.indexed_files:
        st.markdown("**📂 Indexed files:**")
        for fname, size_mb in st.session_state.indexed_files:
            st.markdown(f"- `{fname}` ({size_mb:.1f} MB)")
    else:
        st.markdown("*No files indexed yet.*")

    st.markdown("---")
    st.markdown("**Limits:**")
    st.markdown(f"- Max {MAX_FILES} files")
    st.markdown(f"- Max {MAX_FILE_SIZE_MB} MB per file")
    st.markdown(f"- Max {MAX_TOTAL_SIZE_MB} MB total")
    st.markdown("---")
    st.markdown("**If you hit quota limits:**")
    st.markdown("- App auto-retries and switches models")
    st.markdown("- Or wait a few minutes and retry")
    st.markdown("- [Check your quota](https://ai.dev/rate-limit)")

# ── Validation ────────────────────────────────────────────────────────────────
def validate_files(uploaded_files):
    errors     = []
    total_size = 0

    if len(uploaded_files) > MAX_FILES:
        errors.append(
            f"Too many files — maximum is {MAX_FILES}, "
            f"you uploaded {len(uploaded_files)}."
        )

    for f in uploaded_files:
        size_mb     = len(f.getvalue()) / (1024 * 1024)
        total_size += size_mb
        if size_mb > MAX_FILE_SIZE_MB:
            errors.append(
                f"**{f.name}** is {size_mb:.1f} MB — "
                f"exceeds the {MAX_FILE_SIZE_MB} MB per-file limit."
            )

    if total_size > MAX_TOTAL_SIZE_MB:
        errors.append(
            f"Total upload is {total_size:.1f} MB — "
            f"exceeds the {MAX_TOTAL_SIZE_MB} MB total limit."
        )

    return errors, total_size

# ── PDF processing ────────────────────────────────────────────────────────────
def process_files(uploaded_files):
    os.makedirs("tempDir", exist_ok=True)
    all_docs    = []
    processed   = []
    file_errors = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    progress = st.progress(0, text="Processing files...")

    for i, f in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files), text=f"Processing {f.name}...")
        try:
            pdf_path = f"tempDir/{f.name}"
            with open(pdf_path, "wb") as out:
                out.write(f.getbuffer())

            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()

            for page in pages:
                page.metadata["source_file"] = f.name

            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)

            size_mb = len(f.getvalue()) / (1024 * 1024)
            processed.append((f.name, size_mb, len(chunks)))

        except Exception as e:
            file_errors.append(f"Failed to process **{f.name}**: {str(e)}")

    progress.progress(1.0, text="Building vector index...")

    retriever   = None
    vectorstore = None

    if all_docs:
        # Close + delete old ChromaDB before creating new one
        safe_delete_chroma(st.session_state.chroma_dir)

        # Unique directory per session — avoids Windows file lock conflicts
        new_dir = f"chroma_db_{uuid.uuid4().hex[:8]}"
        st.session_state.chroma_dir = new_dir

        embeddings  = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=new_dir
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )

    progress.empty()
    return retriever, vectorstore, processed, file_errors

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    help=f"Max {MAX_FILES} files · {MAX_FILE_SIZE_MB} MB per file · {MAX_TOTAL_SIZE_MB} MB total"
)

if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    indexed_names = sorted([f for f, _ in st.session_state.indexed_files])

    if current_names != indexed_names:
        errors, total_size = validate_files(uploaded_files)

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()

        st.info(
            f"📦 {len(uploaded_files)} file(s) · "
            f"{total_size:.1f} MB total · "
            f"Limit: {MAX_TOTAL_SIZE_MB} MB"
        )

        retriever, vectorstore, processed, file_errors = process_files(uploaded_files)

        for err in file_errors:
            st.warning(f"⚠️ {err}")

        if retriever:
            st.session_state.retriever     = retriever
            st.session_state.vectorstore   = vectorstore
            st.session_state.indexed_files = [(name, size) for name, size, _ in processed]
            st.session_state.messages      = []

            st.success(f"✅ Indexed {len(processed)} file(s) successfully!")
            with st.expander("📄 Indexing details"):
                for name, size_mb, chunks in processed:
                    st.markdown(f"- **{name}** — {size_mb:.1f} MB · {chunks} chunks")
        else:
            st.error("❌ No documents could be processed.")

# ── Size usage bar ────────────────────────────────────────────────────────────
if uploaded_files:
    total_size = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
    pct        = min(total_size / MAX_TOTAL_SIZE_MB, 1.0)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(pct)
    with col2:
        icon = "🟢" if pct < 0.75 else ("🟡" if pct < 1.0 else "🔴")
        st.markdown(f"{icon} **{total_size:.1f} / {MAX_TOTAL_SIZE_MB} MB**")

st.markdown("---")

# ── Retry delay parser ────────────────────────────────────────────────────────
def parse_retry_delay(error_str: str) -> int:
    match = re.search(r'retry[^\d]*(\d+)', error_str, re.IGNORECASE)
    return int(match.group(1)) if match else 60

# ── Ask Gemini with retry + model fallback ────────────────────────────────────
def ask_gemini(question: str, retriever) -> tuple[str, list]:
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source_file', 'unknown')}, "
        f"Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.
If the answer spans multiple documents, synthesize the information clearly.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

    # Build fallback order: active model first, then remaining preferred models
    try:
        available = {
            m.name.replace("models/", "")
            for m in client.models.list()
        }
    except Exception:
        available = set(PREFERRED_MODELS)

    current       = st.session_state.active_model
    fallback_order = [current] + [
        m for m in PREFERRED_MODELS
        if m != current and m in available
    ]

    last_error = ""

    for attempt_model in fallback_order:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=attempt_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=1024,
                    )
                )

                # Success — update active model if it changed
                if attempt_model != st.session_state.active_model:
                    st.session_state.active_model = attempt_model
                    st.toast(
                        f"Switched to model: `{attempt_model}`",
                        icon="🔄"
                    )

                return response.text, relevant_docs

            except Exception as e:
                last_error   = str(e)
                is_quota     = "429" in last_error or "RESOURCE_EXHAUSTED" in last_error
                is_not_found = "404" in last_error or "NOT_FOUND" in last_error

                if is_not_found:
                    # Model inaccessible — skip to next immediately
                    break

                if is_quota and attempt < 2:
                    delay = min(parse_retry_delay(last_error), 65)
                    with st.spinner(
                        f"⏳ Rate limit on `{attempt_model}` — "
                        f"waiting {delay}s before retry "
                        f"({attempt + 1}/3)..."
                    ):
                        time.sleep(delay)
                    continue

                # Quota fully exhausted for this model — try next
                break

    # All models exhausted
    delay = parse_retry_delay(last_error)
    raise Exception(
        f"All available models hit their quota limits.\n\n"
        f"**Options:**\n"
        f"- Wait **{delay} seconds** and try again\n"
        f"- Your daily free quota may be exhausted — try again tomorrow\n"
        f"- Add billing to your Google AI account for higher limits: "
        f"https://ai.google.dev/gemini-api/docs/rate-limits\n\n"
        f"Last error: `{last_error[:300]}`"
    )

# ── Chat UI ───────────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    file = src.metadata.get("source_file", "unknown")
                    page = src.metadata.get("page", 0) + 1
                    st.markdown(f"**{file} — Page {page}**")
                    st.caption(src.page_content[:300] + "...")
                    st.divider()

if st.session_state.retriever is not None:
    user_question = st.chat_input("Ask a question across all uploaded PDFs...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        answer  = ""
        sources = []

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    answer, sources = ask_gemini(
                        user_question,
                        st.session_state.retriever
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources"):
                            for src in sources:
                                file = src.metadata.get("source_file", "unknown")
                                page = src.metadata.get("page", 0) + 1
                                st.markdown(f"**{file} — Page {page}**")
                                st.caption(src.page_content[:300] + "...")
                                st.divider()
                except Exception as e:
                    answer = str(e)
                    st.error(answer, icon="❌")

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources
        })

else:
    st.info("👆 Upload one or more PDFs above to start chatting.")