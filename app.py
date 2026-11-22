import numpy as np
import PyPDF2
import faiss
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer


APP_NAME = "Intelligent Document Question Answering System"
APP_SUBTITLE = "using RAG and LLM"
APP_SHORT_NAME = "Document Q&A"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


st.set_page_config(
    page_title=f"{APP_NAME} {APP_SUBTITLE}",
    page_icon="◼",
    layout="wide",
)


THEMES = {
    "Light": {
        "bg": "#ffffff",
        "sidebar": "#f7f7f8",
        "surface": "#f4f4f5",
        "border": "#e5e5e5",
        "text": "#171717",
        "muted": "#737373",
        "btn_bg": "#171717",
        "btn_text": "#ffffff",
        "btn_hover": "#404040",
    },
    "Dark": {
        "bg": "#0a0a0a",
        "sidebar": "#0f0f0f",
        "surface": "#171717",
        "border": "#262626",
        "text": "#ededed",
        "muted": "#8a8a8a",
        "btn_bg": "#ffffff",
        "btn_text": "#0a0a0a",
        "btn_hover": "#d4d4d4",
    },
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1100px; }

header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4, h5, p, label, span, li, div { color: var(--text); }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
.stCaption, small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

.stApp .hero {
    font-size: 3.4rem; font-weight: 700; letter-spacing: -0.03em;
    line-height: 1.12; margin: 0.5rem 0 0.8rem 0; color: var(--text);
}
.stApp .hero span { color: var(--muted); }
.stApp .hero-desc {
    font-size: 1.2rem; color: var(--muted); margin: 0 0 2.2rem 0;
}

.stApp p, .stApp li, .stApp label { font-size: 1.05rem; }
.stApp h4 { font-size: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.1rem; }
.stTextInput input, .stTextArea textarea { font-size: 1.05rem; }
[data-testid="stMetricValue"] { font-size: 2rem; }
[data-testid="stMetricLabel"] p { font-size: 0.95rem; }
section[data-testid="stSidebar"] h3 { font-size: 1.4rem; }
.app-footer { font-size: 0.95rem; }

section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0;
    padding: 8px 16px; color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--text) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--text); }

.stButton > button {
    background: var(--btn_bg); color: var(--btn_text);
    border: none; border-radius: 999px;
    padding: 0.55rem 1.4rem; font-weight: 600;
    transition: background 0.15s ease;
}
.stButton > button:hover { background: var(--btn_hover); color: var(--btn_text); }

.stTextInput input, .stTextArea textarea {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 14px !important;
    padding: 0.75rem 1rem;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--text) !important; box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface); border: 1.5px dashed var(--border);
    border-radius: 18px; padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: var(--btn_bg); border: none; border-radius: 999px;
    padding: 0.5rem 1.4rem;
}
[data-testid="stFileUploaderDropzone"] button * { color: var(--btn_text) !important; font-weight: 600; }

[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-weight: 600; }

[data-testid="stAlert"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px;
}

[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px; border-color: var(--border);
    background: var(--surface);
}

pre, code { background: var(--surface) !important; border-radius: 10px; }

.app-footer {
    text-align: center; color: var(--muted) !important;
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid var(--border);
}
"""


# Builds the full CSS for the chosen theme (Light or Dark) by turning the
# colour palette into CSS variables and adding the shared styles.
def build_css(theme_name):
    palette = THEMES[theme_name]
    variables = "\n".join(f"    --{key}: {value};" for key, value in palette.items())
    return f"<style>\n:root {{\n{variables}\n}}\n{BASE_CSS}\n</style>"


# Reads every page of the uploaded PDF and returns all the text as one string,
# after removing invalid characters that would crash the embedding model.
def extract_text_from_pdf(pdf_file):
    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as error:
        st.error(f"Error while reading PDF: {error}")

    text = text.replace("\x00", " ")
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    return text


# Splits the PDF text into small overlapping chunks so that each piece can be
# searched and sent to the LLM separately.
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# Loads the Sentence Transformer model once and keeps it in memory, so it is
# not reloaded every time the page refreshes.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# Converts all chunks into number vectors (embeddings) and stores them in a
# FAISS index so that similar text can be searched quickly.
def create_vector_database(chunks, embedding_model):
    chunks = [str(chunk) for chunk in chunks]
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


# Converts the user's question into a vector and finds the chunks from the PDF
# that are most similar to it.
def search_relevant_chunks(question, chunks, index, embedding_model, top_k=TOP_K):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    top_k = min(top_k, len(chunks))
    _, indices = index.search(question_embedding, top_k)

    relevant_chunks = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            relevant_chunks.append(chunks[i])

    return relevant_chunks


# Sends the question and the matched PDF text to the local Ollama LLM and
# returns the answer it generates (only from the given context).
def ask_llm(question, context):
    prompt = f"""
You are a document question answering assistant.

Instructions:
1. Answer only from the given context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Give a clear and simple explanation.
5. Use bullet points when helpful.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        return f"Ollama is not running. Start it with: ollama run {OLLAMA_MODEL}"

    except Exception as error:
        return f"Error while communicating with the LLM: {error}"


# Shows the step-by-step RAG pipeline as a text diagram in the Project Flow tab.
def show_project_flow():
    st.markdown(
        """
```text
PDF Upload
    ↓
Extract Text
    ↓
Split into Chunks
    ↓
Create Embeddings
    ↓
Store in FAISS Vector Database
    ↓
User Asks a Question
    ↓
Embed the Question
    ↓
Retrieve Similar Chunks
    ↓
Build Prompt (Context + Question)
    ↓
LLM Generates the Answer
```
"""
    )


# Extracts, chunks and indexes the uploaded PDF once, then saves the result in
# session state so asking more questions does not repeat this work.
def process_pdf(uploaded_file, embedding_model):
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.get("file_key") == file_key:
        return st.session_state["processed"]

    with st.spinner("Reading PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if not pdf_text.strip():
        return None

    with st.spinner("Splitting text into chunks..."):
        chunks = split_text_into_chunks(pdf_text)

    with st.spinner("Building the vector database..."):
        index, _ = create_vector_database(chunks, embedding_model)

    processed = {"text": pdf_text, "chunks": chunks, "index": index}
    st.session_state["file_key"] = file_key
    st.session_state["processed"] = processed
    return processed


with st.sidebar:
    st.markdown(f"### ◼ {APP_SHORT_NAME}")
    st.caption("RAG + LLM")

    theme_choice = st.radio("Appearance", ["Light", "Dark"], horizontal=True)

    st.divider()

    st.markdown("**Stack**")
    st.caption("Python · Streamlit · PyPDF2 · Sentence Transformers · FAISS · Ollama")

    st.markdown("**Start the model**")
    st.code(f"ollama run {OLLAMA_MODEL}", language="bash")

st.markdown(build_css(theme_choice), unsafe_allow_html=True)

st.markdown(
    f'<div class="hero">{APP_NAME} <span>{APP_SUBTITLE}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-desc">Upload a PDF, ask a question, and get an answer grounded in its content.</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_flow, tab_concepts = st.tabs(
    ["Upload PDF", "Project Flow", "Technical Concepts"]
)

with tab_upload:
    uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"])

    embedding_model = load_embedding_model()

    if uploaded_file is not None:
        data = process_pdf(uploaded_file, embedding_model)

        if data is None:
            st.error("No readable text found in this PDF (it may be a scanned image).")
        else:
            st.success("PDF processed. You can now ask questions.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Characters", f"{len(data['text']):,}")
            col2.metric("Chunks", len(data["chunks"]))
            col3.metric("Chunk size", f"{CHUNK_SIZE} chars")

            with st.expander("Preview extracted text"):
                st.text_area(
                    "First 2000 characters",
                    data["text"][:2000],
                    height=220,
                    label_visibility="collapsed",
                )

            st.markdown("#### Ask a question")
            question = st.text_input(
                "Question",
                placeholder="e.g. What is the main topic of this document?",
                label_visibility="collapsed",
            )

            if question:
                with st.spinner("Searching the document..."):
                    relevant_chunks = search_relevant_chunks(
                        question, data["chunks"], data["index"], embedding_model
                    )
                context = "\n\n".join(relevant_chunks)

                with st.spinner("Generating answer..."):
                    answer = ask_llm(question, context)

                with st.container(border=True):
                    st.markdown("**Answer**")
                    st.markdown(answer)

                with st.expander("View retrieved context"):
                    st.write(context)

with tab_flow:
    st.markdown("#### RAG architecture flow")
    show_project_flow()

with tab_concepts:
    st.markdown("#### Technical concepts")
    st.markdown(
        """
**1. PDF processing** – PyPDF2 extracts text from every page.

**2. Text chunking** – long text is split into small overlapping pieces.

**3. Embeddings** – Sentence Transformers turn each chunk into a numeric vector.

**4. Vector database** – FAISS stores the vectors and finds similar ones quickly.

**5. Semantic search** – the question is embedded and matched against the chunks.

**6. RAG** – relevant context is retrieved first, then given to the LLM.

**7. LLM** – Llama 3 running locally through Ollama writes the final answer.

**8. Prompt engineering** – context and question are combined in a structured prompt.
"""
    )

st.markdown(
    f'<div class="app-footer">{APP_NAME} {APP_SUBTITLE} · Built with Streamlit, FAISS and Ollama</div>',
    unsafe_allow_html=True,
)
