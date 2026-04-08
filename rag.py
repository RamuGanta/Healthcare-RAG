from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import warnings
warnings.filterwarnings("ignore")

# Scrape CDC pages (replacing your hardcoded docs)
urls = [
    # Heart, Blood & Stroke
    "https://www.cdc.gov/heart-disease/about/index.html",
    "https://www.cdc.gov/stroke/about/index.html",
    "https://www.cdc.gov/high-blood-pressure/about/index.html",
    "https://www.cdc.gov/cholesterol/about/index.html",
    "https://www.cdc.gov/atrial-fibrillation/about/index.html",
    "https://www.cdc.gov/heart-failure/about/index.html",
    "https://www.cdc.gov/blood-clots/about/index.html",
    "https://www.cdc.gov/blood-disorders/about/index.html",
    "https://www.cdc.gov/sickle-cell/about/index.html",

    # Cancer
    "https://www.cdc.gov/cancer/about/index.html",
    "https://www.cdc.gov/breast-cancer/about/index.html",
    "https://www.cdc.gov/lung-cancer/about/index.html",
    "https://www.cdc.gov/colorectal-cancer/about/index.html",
    "https://www.cdc.gov/skin-cancer/about/index.html",
    "https://www.cdc.gov/prostate-cancer/about/index.html",
    "https://www.cdc.gov/cervical-cancer/about/index.html",
    "https://www.cdc.gov/cancer-screening/about/index.html",

    # Respiratory
    "https://www.cdc.gov/asthma/about/index.html",
    "https://www.cdc.gov/copd/about/index.html",
    "https://www.cdc.gov/flu/about/index.html",
    "https://www.cdc.gov/covid/about/index.html",
    "https://www.cdc.gov/pneumonia/about/index.html",
    "https://www.cdc.gov/lung-disease/about/index.html",

    # Diabetes
    "https://www.cdc.gov/diabetes/about/index.html",
    "https://www.cdc.gov/diabetes/about/diabetes-type-1.html",
    "https://www.cdc.gov/diabetes/about/diabetes-type-2.html",
    "https://www.cdc.gov/diabetes/about/diabetes-prevention.html",

    # Mental Health
    "https://www.cdc.gov/mental-health/about/index.html",
    "https://www.cdc.gov/depression/about/index.html",
    "https://www.cdc.gov/anxiety/about/index.html",
    "https://www.cdc.gov/suicide/about/index.html",
    "https://www.cdc.gov/adhd/about/index.html",

    # Nutrition & Lifestyle
    "https://www.cdc.gov/nutrition/about/index.html",
    "https://www.cdc.gov/healthy-weight/about/index.html",
    "https://www.cdc.gov/obesity/about/index.html",
    "https://www.cdc.gov/obesity/about/causes.html",
    "https://www.cdc.gov/physical-activity/about/index.html",

    # Kidney & Liver
    "https://www.cdc.gov/kidney-disease/about/index.html",
    "https://www.cdc.gov/liver-disease/about/index.html",

    # Infections & Vaccines
    "https://www.cdc.gov/hiv/about/index.html",
    "https://www.cdc.gov/hepatitis/about/index.html",
    "https://www.cdc.gov/sti/about/index.html",
    "https://www.cdc.gov/vaccines/about/index.html",
    "https://www.cdc.gov/mrsa/about/index.html",
    "https://www.cdc.gov/tb/about/index.html",

    # Other
    "https://www.cdc.gov/arthritis/about/index.html",
    "https://www.cdc.gov/alzheimers-dementia/about/index.html",
    "https://www.cdc.gov/epilepsy/about/index.html",
    "https://www.cdc.gov/sleep/about/index.html",
    "https://www.cdc.gov/oral-health/about/index.html",
    "https://www.cdc.gov/vision-health/about/index.html",
]

docs = []
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Find where real content starts
    start = text.find("Key points")
    if start == -1:
        start = text.find("About")
    if start == -1:
        start = 500
    text = text[start:]
    docs.append(text[:1500])  # first 1000 chars per doc

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_vectors = embedder.encode(docs)

question = "what causes lung cancer"
q_vector = embedder.encode(question)

scores = np.dot(doc_vectors, q_vector)
best_doc = docs[scores.argmax()]

best_score = scores.max()
print(f"Relevance score: {best_score:.2f}")

if best_score < 0.35:
    print("Sorry, I don't have enough relevant information to answer that question.")
else:
    best_doc = docs[scores.argmax()]
    print("Best doc:", best_doc[:300])

    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    prompt = f"Context: {best_doc}\nQuestion: {question}\nAnswer:"
    answer = generator(prompt, max_new_tokens=50)
    print(answer[0]["generated_text"])
