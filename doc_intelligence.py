import streamlit as st
import pandas as pd
import pdfplumber
import ssl
import numpy as np
import cv2
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from sentence_transformers import SentenceTransformer, util
import re
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_PATH = os.path.abspath("/Users/Jules.Gerard/Downloads/all-MiniLM-L6-v2")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

st_model = SentenceTransformer(MODEL_PATH)

AZURE_ENDPOINT = "https://menu-ai.cognitiveservices.azure.com/"
AZURE_KEY = "<your-azure-key>"  # Replace with secure retrieval in production
doc_client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

st.set_page_config(page_title="SGEYE: AI-Powered Smart Extraction & Classification", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: white; }
    .stApp { background-color: #121212; }
    .stButton>button { width: 100%; font-size: 18px; font-weight: bold; padding: 10px; }
    h1, h2, h3 { color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    st.image("sgeye.jpg", width=110)
with col2:
    st.video("/Users/Jules.Gerard/Downloads/firefly.mp4", start_time=0, loop=True)

st.title("🚀 SGEYE: Smart Extraction, Pricing Detection, and Context Classification")

files = st.sidebar.file_uploader("📥 Upload Menus (PDF)", type=["pdf"], accept_multiple_files=True)
brand_file = st.sidebar.file_uploader("📊 Upload Brand List (CSV)", type=["csv"])
threshold = st.sidebar.slider("🔎 Match Sensitivity Threshold", 0, 100, 75, 1)

def azure_extract_text_and_prices(pdf_file):
    pdf_bytes = pdf_file.read()
    poller = doc_client.begin_analyze_document(
        model_id="prebuilt-layout", document=pdf_bytes, content_type="application/pdf")
    result = poller.result()

    lines_with_prices = []
    for page in result.pages:
        for line in page.lines:
            text = line.content
            price_match = re.search(r'\$?\d+\.\d{2}', text)
            price = price_match.group(0) if price_match else None
            lines_with_prices.append({"line": text, "price": price})

    return lines_with_prices

def classify_context(line):
    cocktail_keywords = ["shake", "muddle", "stir", "garnish", "pour"]
    if any(word in line.lower() for word in cocktail_keywords):
        return "Cocktail Recipe"
    return "Stand-alone Menu Item"

def match_proper_noun_items(menu_lines, brand_names, embeddings, threshold):
    matches = []
    used_brands = set()
    line_texts = [entry["line"] for entry in menu_lines]
    item_embeddings = st_model.encode(line_texts, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(item_embeddings, embeddings).cpu().numpy()

    for idx, entry in enumerate(menu_lines):
        line = entry["line"]
        price = entry["price"]
        context_type = classify_context(line)
        if not any(w[0].isupper() for w in line.split()):
            matches.append([line, "No Match", 0, price, context_type])
            continue

        best_idx = np.argmax(sims[idx])
        score = sims[idx][best_idx] * 100
        brand = brand_names[best_idx]

        if brand not in used_brands and score >= threshold:
            used_brands.add(brand)
            matches.append([line, brand, int(score), price, context_type])
        else:
            matches.append([line, "No Match", 0, price, context_type])

    df = pd.DataFrame(matches, columns=["Line Text", "Matched Brand", "Score", "Detected Price", "Context Type"])
    return df

if files and brand_file:
    st.sidebar.success("✅ Files and brand list uploaded!")

    brands_df = pd.read_csv(brand_file)
    brand_names = brands_df["brand_name"].tolist()
    brand_emb = st_model.encode(brand_names, convert_to_tensor=True)

    all_results = []

    for file in files:
        st.write(f"🔎 Processing {file.name} with Azure Document Intelligence...")
        extracted_lines = azure_extract_text_and_prices(file)

        matched_df = match_proper_noun_items(extracted_lines, brand_names, brand_emb, threshold)
        merged = matched_df.merge(brands_df, left_on="Matched Brand", right_on="brand_name", how="left")
        merged.insert(0, "Source File", file.name)

        all_results.append(merged)

    final_results = pd.concat(all_results).sort_values(by="Score", ascending=False)
    st.subheader("🔎 Detailed Matching, Pricing, and Context Analysis Results")
    st.dataframe(final_results, use_container_width=True)