import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="HealthGuard AI", page_icon="🏥")
st.title("🏥 HealthGuard AI")
st.caption("Ask health questions — powered by CDC documents")

# ─── Load docs (cached so it only runs once) ───
@st.cache_data(show_spinner="📄 Scraping CDC documents...")
def load_docs():
    urls = [
        "https://www.cdc.gov/heart-disease/about/index.html",
        "https://www.cdc.gov/stroke/about/index.html",
        "https://www.cdc.gov/high-blood-pressure/about/index.html",
        "https://www.cdc.gov/cholesterol/about/index.html",
        "https://www.cdc.gov/atrial-fibrillation/about/index.html",
        "https://www.cdc.gov/heart-failure/about/index.html",
        "https://www.cdc.gov/blood-clots/about/index.html",
        "https://www.cdc.gov/blood-disorders/about/index.html",
        "https://www.cdc.gov/sickle-cell/about/index.html",
        "https://www.cdc.gov/cancer/about/index.html",
        "https://www.cdc.gov/breast-cancer/about/index.html",
        "https://www.cdc.gov/lung-cancer/about/index.html",
        "https://www.cdc.gov/colorectal-cancer/about/index.html",
        "https://www.cdc.gov/skin-cancer/about/index.html",
        "https://www.cdc.gov/prostate-cancer/about/index.html",
        "https://www.cdc.gov/cervical-cancer/about/index.html",
        "https://www.cdc.gov/cancer-screening/about/index.html",
        "https://www.cdc.gov/asthma/about/index.html",
        "https://www.cdc.gov/copd/about/index.html",
        "https://www.cdc.gov/flu/about/index.html",
        "https://www.cdc.gov/covid/about/index.html",
        "https://www.cdc.gov/pneumonia/about/index.html",
        "https://www.cdc.gov/lung-disease/about/index.html",
        "https://www.cdc.gov/diabetes/about/index.html",
        "https://www.cdc.gov/diabetes/about/diabetes-type-1.html",
        "https://www.cdc.gov/diabetes/about/diabetes-type-2.html",
        "https://www.cdc.gov/diabetes/about/diabetes-prevention.html",
        "https://www.cdc.gov/mental-health/about/index.html",
        "https://www.cdc.gov/depression/about/index.html",
        "https://www.cdc.gov/anxiety/about/index.html",
        "https://www.cdc.gov/suicide/about/index.html",
        "https://www.cdc.gov/adhd/about/index.html",
        "https://www.cdc.gov/nutrition/about/index.html",
        "https://www.cdc.gov/healthy-weight/about/index.html",
        "https://www.cdc.gov/obesity/about/index.html",
        "https://www.cdc.gov/obesity/about/causes.html",
        "https://www.cdc.gov/physical-activity/about/index.html",
        "https://www.cdc.gov/kidney-disease/about/index.html",
        "https://www.cdc.gov/liver-disease/about/index.html",
        "https://www.cdc.gov/hiv/about/index.html",
        "https://www.cdc.gov/hepatitis/about/index.html",
        "https://www.cdc.gov/sti/about/index.html",
        "https://www.cdc.gov/vaccines/about/index.html",
        "https://www.cdc.gov/mrsa/about/index.html",
        "https://www.cdc.gov/tb/about/index.html",
        "https://www.cdc.gov/arthritis/about/index.html",
        "https://www.cdc.gov/alzheimers-dementia/about/index.html",
        "https://www.cdc.gov/epilepsy/about/index.html",
        "https://www.cdc.gov/sleep/about/index.html",
        "https://www.cdc.gov/oral-health/about/index.html",
        "https://www.cdc.gov/vision-health/about/index.html",
    ]

    docs = []
    doc_urls = []
    for url in urls:
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            start = text.find("Key points")
            if start == -1:
                start = text.find("About")
            if start == -1:
                start = 500
            text = text[start:]
            docs.append(text[:1500])
            doc_urls.append(url)
        except:
            continue
    return docs, doc_urls

@st.cache_resource(show_spinner="🧠 Loading AI models...")
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, generator

# Load everything
docs, doc_urls = load_docs()
embedder, generator = load_models()
doc_vectors = embedder.encode(docs)

st.success(f"✅ Loaded {len(docs)} CDC documents")

# ─── Chat ───
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a health question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        q_vector = embedder.encode(question)
        scores = np.dot(doc_vectors, q_vector)
        best_score = scores.max()

        if best_score < 0.35:
            answer = "Sorry, I don't have enough relevant information to answer that question. Try asking about a specific health condition."
            st.markdown(answer)
        else:
            best_doc = docs[scores.argmax()]
            best_url = doc_urls[scores.argmax()]

            prompt = f"Context: {best_doc}\nQuestion: {question}\nAnswer:"
            result = generator(prompt, max_new_tokens=50)
            answer = result[0]["generated_text"]

            st.markdown(answer)
            st.markdown(f"**Relevance:** {best_score:.0%}")
            st.markdown(f"**Source:** [CDC Link]({best_url})")

        st.session_state.messages.append({"role": "assistant", "content": answer})