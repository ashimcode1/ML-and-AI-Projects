import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
NOT_FOUND = "I couldn't find this in the document."


@st.cache_resource
def load_embedder():
    return TextEmbedding(EMBED_MODEL)


@st.cache_resource
def load_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


def embed(embedder, texts):
    vectors = np.array(list(embedder.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def read_pdf(file):
    pages = []
    for number, page in enumerate(PdfReader(file).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def make_chunks(pages):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for number, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"page": number, "text": piece})
    return chunks


def retrieve(question, embedder, vectors, chunks):
    query = embed(embedder, [question])[0]
    scores = vectors @ query
    top = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in top]


def answer_question(question, retrieved, client):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in retrieved)
    prompt = (
        "Answer the question using only the context below.\n"
        f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}"\n'
        "Mention the page number(s) you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    st.set_page_config(page_title="Document Assistant", page_icon="📄")
    st.title("📄 Document Assistant")

    client = load_client()
    if client is None:
        st.error("GROQ_API_KEY not found. Add it to a .env file next to app.py.")
        st.stop()

    embedder = load_embedder()
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf is None:
        return

    file_key = f"{pdf.name}-{pdf.size}"
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Processing PDF..."):
            chunks = make_chunks(read_pdf(pdf))
            if not chunks:
                st.error("No text found. The PDF may be scanned images.")
                st.stop()
            vectors = embed(embedder, [c["text"] for c in chunks])
        st.session_state.update(file_key=file_key, chunks=chunks, vectors=vectors)

    chunks = st.session_state["chunks"]
    vectors = st.session_state["vectors"]
    st.success(f"Processed: {len(chunks)} chunks")

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Finding answer..."):
            try:
                retrieved = retrieve(question, embedder, vectors, chunks)
                answer = answer_question(question, retrieved, client)
            except Exception as error:
                st.error(f"Error: {error}")
                st.stop()

        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources"):
            for c in retrieved:
                st.markdown(f"**Page {c['page']}**")
                st.write(c["text"])


if __name__ == "__main__":
    main()