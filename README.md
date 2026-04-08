# 🏥 HealthGuard AI

**A Retrieval-Augmented Generation (RAG) system that answers health questions using 50+ official CDC documents.**

Built entirely with open-source models — no API keys, no paid services, no cloud dependencies.

---

## 🎯 What This Project Does

Ask a health question → the system searches through 50+ CDC health pages → finds the most relevant document → generates an answer using that context → links you to the original CDC source.

If no document is relevant enough (score < 0.35), it tells you honestly instead of making something up.

```
"What causes lung cancer?"

→ Retrieves: CDC Lung Cancer page (relevance: 63%)
→ Answer: Cigarette smoking
→ Source: https://www.cdc.gov/lung-cancer/about/index.html
```

---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
  User Question ──→ │   SentenceTransformer   │──→ Query Vector
                    │   (all-MiniLM-L6-v2)    │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Cosine Similarity      │──→ Ranked Documents
                    │   (NumPy dot product)    │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Relevance Filter       │──→ Score ≥ 0.35?
                    │   (Threshold Gate)       │    No → "Not enough info"
                    └─────────────────────────┘
                              │ Yes
                              ▼
                    ┌─────────────────────────┐
                    │   Flan-T5-Small          │──→ Generated Answer
                    │   (Text Generation)      │    + CDC Source Link
                    └─────────────────────────┘
```

---

## 📋 Features

- **50+ CDC health topics** scraped and cleaned at runtime — heart disease, cancer, diabetes, mental health, infections, and more
- **Semantic search** using sentence embeddings — finds relevant docs by meaning, not just keywords
- **Relevance thresholding** — rejects out-of-scope questions instead of hallucinating answers
- **Source attribution** — every answer includes a direct link to the CDC source page
- **Zero cost** — runs on open-source models, no API keys or paid services needed
- **Chat interface** — Streamlit-based conversational UI with message history

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | `all-MiniLM-L6-v2` | Fast, accurate sentence embeddings (384 dimensions) |
| Vector Search | NumPy `dot` | Simple cosine similarity — no external vector DB needed |
| Generator | `google/flan-t5-small` | Lightweight text generation that runs on CPU |
| Web Scraping | `requests` + `BeautifulSoup` | Extracts clean text from CDC HTML pages |
| Frontend | Streamlit | Chat UI with caching and session state |
| Data Source | CDC.gov | Authoritative public health information |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/RamuGanta/Healthcare-RAG.git
cd Healthcare-RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will scrape CDC pages on first load (takes ~30 seconds), then cache everything for subsequent runs.

### Or run the terminal version
```bash
python rag.py
```

---

## 📁 Project Structure

```
Healthcare-RAG/
├── app.py               # Streamlit chat app — full RAG pipeline with UI
├── rag.py               # Terminal version — same pipeline, no UI
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🩺 Health Topics Covered (50+)

| Category | Topics |
|----------|--------|
| **Heart & Blood** | Heart Disease, Stroke, High Blood Pressure, Cholesterol, Atrial Fibrillation, Heart Failure, Blood Clots, Blood Disorders, Sickle Cell |
| **Cancer** | Breast, Lung, Colorectal, Skin, Prostate, Cervical, Cancer Screening |
| **Respiratory** | Asthma, COPD, Flu, COVID-19, Pneumonia, Lung Disease |
| **Diabetes** | Type 1, Type 2, Prevention |
| **Mental Health** | Depression, Anxiety, ADHD, Suicide Prevention |
| **Infections** | HIV, Hepatitis, STIs, TB, MRSA, Vaccines |
| **Nutrition** | Nutrition, Healthy Weight, Obesity, Physical Activity |
| **Other** | Kidney Disease, Liver Disease, Alzheimer's, Arthritis, Epilepsy, Sleep, Oral Health, Vision |

---

## 🔍 How the RAG Pipeline Works

**1. Document Ingestion**
- Scrapes 50+ CDC health pages using `requests`
- Parses HTML with `BeautifulSoup`, removes scripts/nav/footer
- Finds actual content start by looking for "Key points" or "About" markers
- Extracts first 1500 characters of clean content per page

**2. Embedding**
- Encodes all document chunks using `all-MiniLM-L6-v2` (SentenceTransformer)
- Encodes the user question with the same model
- Both produce 384-dimensional dense vectors

**3. Retrieval**
- Computes cosine similarity between query vector and all document vectors using `np.dot`
- Selects the highest-scoring document
- If score < 0.35 → rejects the query as out-of-scope

**4. Generation**
- Passes retrieved context + question to `Flan-T5-Small`
- Prompt format: `Context: {doc}\nQuestion: {query}\nAnswer:`
- Returns generated answer + CDC source URL

---

## 📊 Example Queries

| Question | Matched Topic | Score | Answer |
|----------|--------------|-------|--------|
| "What causes lung cancer?" | Lung Cancer | 0.63 | Cigarette smoking |
| "How to prevent heart attack?" | Heart Disease | 0.60 | Know your risk for heart disease |
| "What to eat to live longer?" | — | 0.23 | *Rejected: below threshold* |
| "Healthy food for heart health" | Cholesterol | 0.58 | Blood cholesterol is essential for good health... |

---

## 🧠 What I Learned Building This

- **RAG from scratch** — document ingestion, chunking, embedding, retrieval, and generation without using LangChain or any RAG framework
- **Web scraping & cleaning** — extracting useful content from HTML while removing navigation, banners, and boilerplate
- **Semantic search** — how cosine similarity on sentence embeddings finds relevant documents by meaning rather than keywords
- **Relevance thresholding** — preventing hallucinated answers by rejecting low-confidence retrievals
- **Streamlit caching** — using `@st.cache_data` and `@st.cache_resource` to avoid reloading models and re-scraping on every interaction

---

## 🔮 Future Improvements

- [ ] Chunk documents into smaller pieces for more precise retrieval
- [ ] Return top-3 documents instead of just the best match
- [ ] Add RAGAS evaluation metrics (faithfulness, relevance, context recall)
- [ ] Upgrade to a larger generator model (Flan-T5-Base or Flan-T5-Large)
- [ ] Pre-save scraped documents to avoid runtime scraping
- [ ] Deploy to HuggingFace Spaces

---

## ⚕️ Disclaimer

This tool is for **informational and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical questions.

---

Built by [Ramu Ganta](https://github.com/RamuGanta)
