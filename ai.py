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

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_PATH = os.path.abspath("/Users/Jules.Gerard/Downloads/all-MiniLM-L6-v2")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

st_model = SentenceTransformer(MODEL_PATH)

AZURE_ENDPOINT = "https://menu-ai.cognitiveservices.azure.com/"
AZURE_KEY = "<your-azure-key>"  # Replace with secure retrieval in production
doc_client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

st.set_page_config(page_title="SGEYE: Advanced Extraction & Classification", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: #E0E0E0; }
    .stApp { background-color: #121212; }
    .stButton>button { width: 100%; font-size: 18px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: black; border-radius: 8px; }
    h1, h2, h3 { color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    st.image("sgeye.jpg", width=100)
with col2:
    st.video("/Users/Jules.Gerard/Downloads/jules_sgeye.mp4", start_time=0, loop=True)


st.title("🚀 SGEYE")

files = st.sidebar.file_uploader("📥 Upload Menus (PDF)", type=["pdf"], accept_multiple_files=True)
brand_file = st.sidebar.file_uploader("📊 Upload Brand List (CSV)", type=["csv"])
threshold = st.sidebar.slider("🔎 Matching Threshold", 0, 100, 75, 1)

progress_placeholder = st.empty()
def show_temperature_bar(percentage):
    progress_placeholder.progress(percentage / 100.0, text=f"Processing... {percentage}%")

def azure_extract_text_and_prices(pdf_file):
    pdf_bytes = pdf_file.read()
    # Simulated results with extracted price and contextual hints
    return [
        {"line": "Bombay Dry Gin $12.00", "price": "$12.00", "likely_context": "Standalone Item"},
        {"line": "Tito's Handmade Vodka in Cosmopolitan recipe $13.50", "price": "$13.50", "likely_context": "Part of Recipe"},
        {"line": "Veuve Clicquot Champagne Bottle $85.00", "price": "$85.00", "likely_context": "Standalone Bottle"}
    ]

def classify_context(line):
    cocktail_keywords = ["shake", "muddle", "stir", "garnish", "pour", "recipe", "combine"]
    bottle_keywords = ["bottle", "magnum", "750ml", "liter"]
    text_lower = line.lower()
    if any(word in text_lower for word in cocktail_keywords):
        return "Part of Recipe"
    if any(word in text_lower for word in bottle_keywords):
        return "Standalone Bottle"
    return "Standalone Item"

def match_proper_noun_lines(menu_lines, brand_names, embeddings, threshold):
    results = []
    used_brands = set()
    line_texts = [entry["line"] for entry in menu_lines]
    item_emb = st_model.encode(line_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(item_emb, embeddings).cpu().numpy()

    for i, entry in enumerate(menu_lines):
        line = entry["line"]
        price = entry["price"]
        user_context = entry.get("likely_context") or classify_context(line)

        if not any(word[0].isupper() for word in line.split()):
            results.append([line, "No Match", 0, price, user_context])
            continue

        top_idx = np.argmax(similarities[i])
        score = similarities[i][top_idx] * 100
        brand = brand_names[top_idx]

        if brand not in used_brands and score >= threshold:
            used_brands.add(brand)
            results.append([line, brand, int(score), price, user_context])
        else:
            results.append([line, "No Match", 0, price, user_context])

    return pd.DataFrame(results, columns=["Line Text", "Matched Brand", "Score", "Price", "Predicted Context"])

if files:
    for idx, file in enumerate(files):
        show_temperature_bar((idx + 1) * int(50 / len(files)))
        st.write(f"📄 OCR Results with Price & Context for: {file.name}")
        ocr_output = azure_extract_text_and_prices(file)
        st.dataframe(pd.DataFrame(ocr_output), use_container_width=True)

    if brand_file:
        st.sidebar.success("✅ Matching Phase Starting")
        brands_df = pd.read_csv(brand_file)
        brand_names = brands_df["brand_name"].tolist()
        brand_embeddings = st_model.encode(brand_names, convert_to_tensor=True)

        all_results = []
        for idx, file in enumerate(files):
            show_temperature_bar(50 + ((idx + 1) * int(50 / len(files))))
            extracted_lines = azure_extract_text_and_prices(file)
            matched_results = match_proper_noun_lines(extracted_lines, brand_names, brand_embeddings, threshold)
            merged = matched_results.merge(brands_df, left_on="Matched Brand", right_on="brand_name", how="left")
            merged.insert(0, "Source File", file.name)
            all_results.append(merged)

        st.subheader("✅ Matched Results with Price & Context Classification:")
        st.dataframe(pd.concat(all_results).sort_values(by="Score", ascending=False), use_container_width=True)