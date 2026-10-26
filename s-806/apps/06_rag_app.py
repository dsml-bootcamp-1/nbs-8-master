"""Step 6: Day 5's RAG pipeline as an app: ask the EU AI Act, get an answer with its sources.

    streamlit run apps/06_rag_app.py
"""

import os
import re
from collections import Counter

import chromadb
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "gpt-5.4-nano"
NOISE = re.compile(r"^(EN )?OJ L, 12\.7\.2024( EN)?$|^ELI: http")
HEADING = re.compile(r"^(Article \d+|ANNEX [IVXL]+)$")
SYSTEM = """You answer questions about the EU AI Act using ONLY the excerpts provided.
Cite the excerpt ids in square brackets, e.g. [chunk-12]. If the excerpts do not contain the answer,
reply exactly "I don't know based on the provided document." Answer in 2-4 sentences."""


def api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Indexing the AI Act (once)...")
def build_index(path="data/eu_ai_act.pdf", size=150, overlap=30):
    """Day 5's pipeline: load, chunk with metadata, embed, store. Cached: it runs once, not at every rerun."""
    words, section = [], "Recitals"
    for number, page in enumerate(PdfReader(path).pages, start=1):
        for line in page.extract_text().split("\n"):
            line = line.strip()
            if not line or NOISE.search(line):
                continue
            if HEADING.match(line):
                section = line
            words += [(word, number, section) for word in line.split()]
    chunks = []
    for start in range(0, len(words), size - overlap):
        piece = words[start:start + size]
        chunks.append({"id": f"chunk-{len(chunks)}", "text": " ".join(w for w, _, _ in piece), "page": piece[0][1],
                       "section": Counter(s for _, _, s in piece).most_common(1)[0][0]})
        if start + size >= len(words):
            break
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    collection = chromadb.Client().get_or_create_collection("ai_act", configuration={"hnsw": {"space": "cosine"}})
    if collection.count() == 0:
        collection.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                       embeddings=embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False),
                       metadatas=[{"page": c["page"], "section": c["section"]} for c in chunks])
    return collection, embedder


st.title("Ask the EU AI Act")
key = api_key()
if not key:
    st.error("No OpenAI key found. Add it to .streamlit/secrets.toml or .env (see the README).")
    st.stop()

collection, embedder = build_index()
client = OpenAI(api_key=key)

question = st.text_input("Your question", placeholder="Which AI practices are banned?")
k = st.sidebar.slider("Excerpts to retrieve (k)", 1, 10, 5)
rules_only = st.sidebar.checkbox("Articles and annexes only (no recitals)", value=True)

if question:
    result = collection.query(query_embeddings=embedder.encode([question], normalize_embeddings=True), n_results=k,
                              where={"section": {"$ne": "Recitals"}} if rules_only else None)
    excerpts = list(zip(result["ids"][0], result["documents"][0], result["metadatas"][0]))
    context = "\n\n".join(f"[{i}] ({m['section']}, page {m['page']})\n{text}" for i, text, m in excerpts)
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(model=MODEL, messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}])
    answer = response.choices[0].message.content
    st.markdown(answer)
    st.subheader("Sources")
    for i, text, m in excerpts:
        cited = i in answer
        with st.expander(f"{'✅ ' if cited else ''}{i}: {m['section']}, page {m['page']}", expanded=False):
            st.write(text)
    st.caption(f"{response.usage.prompt_tokens:,} input and {response.usage.completion_tokens:,} output tokens. "
               "Source: Regulation (EU) 2024/1689, EUR-Lex. Not legal advice.")
