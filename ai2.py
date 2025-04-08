import streamlit as st
import pandas as pd
import pdfplumber
import easyocr
import ssl
import numpy as np
import cv2
import os
import re
import torch
import sys
from sentence_transformers import SentenceTransformer, util

# 🛡️ Fix torch class inspection bug for Streamlit
try:
    if hasattr(torch, "_classes"):
        torch._classes = sys.modules.get("torch._classes", {})
except Exception:
    pass

ssl._create_default_https_context = ssl._create_unverified_context

# 🌐 Page config + styles
st.set_page_config(page_title="SGEYE: AI Menu Magic", page_icon="📜", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: #FFFFFF; }
    .stApp { background-color: #121212; }
    .stButton>button { width: 100%; font-size: 18px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: black; border-radius: 8px; }
    h1, h2, h3 { color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# 🎥 Video from YouTube (autoplay via embed)
st.markdown("""
<iframe width="100%" height="315" src="https://www.youtube.com/embed/MCjZAtgkrHM?autoplay=1&mute=1&loop=1&playlist=MCjZAtgkrHM" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
""", unsafe_allow_html=True)

st.image("sgeye.jpg", width=80)
st.title("📜 SGEYE: AI Menu Magic")

# 📁 File inputs
files = st.sidebar.file_uploader("🗕️ Upload Menus (PDF, Image, CSV, or XLSX)", type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"], accept_multiple_files=True)
brand_file = st.sidebar.file_uploader("📊 Upload Brand List (CSV or XLSX)", type=["csv", "xlsx"])
threshold = st.sidebar.slider("🔎 Transformer Similarity Threshold", 0, 100, 75, 1)

progress_placeholder = st.empty()

@st.cache_resource
def load_transformer_model():
    local_model_path = os.path.join(os.path.dirname(__file__), "all-MiniLM-L6-v2")
    return SentenceTransformer(local_model_path)

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], model_storage_directory='models', download_enabled=False)

st_model = load_transformer_model()
reader = load_ocr_reader()

def show_temperature_bar(percentage):
    progress_placeholder.progress(percentage / 100.0, text=f"Processing... {percentage}%")

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()

def extract_text_from_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = reader.readtext(image, detail=0)
    return "\n".join(results)

def extract_lines_from_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file).split("\n")
    elif file.type in ["image/png", "image/jpeg"]:
        file.seek(0)
        return extract_text_from_image(file).split("\n")
    elif file.type == "text/csv":
        return pd.read_csv(file).iloc[:, 0].dropna().tolist()
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file).iloc[:, 0].dropna().tolist()
    return []

def classify_context(line):
    cocktail = ["shake", "muddle", "stir", "garnish", "recipe", "pour"]
    bottle = ["bottle", "magnum", "750ml"]
    if any(k in line.lower() for k in cocktail):
        return "Part of Recipe"
    if any(k in line.lower() for k in bottle):
        return "Bottle / Standalone"
    return "Standalone Item"

def extract_price(line):
    match = re.search(r'\$?\d+\.\d{2}', line)
    return match.group(0) if match else ""

def transformer_match(menu_lines, brand_names, brand_embeddings, threshold, src):
    results = []
    lines_emb = st_model.encode(menu_lines, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(lines_emb, brand_embeddings).cpu().numpy()

    used = set()
    for i, line in enumerate(menu_lines):
        row = sim[i]
        top = np.argmax(row)
        score = row[top] * 100
        brand = brand_names[top]
        price = extract_price(line)
        context = classify_context(line)

        if brand not in used and score >= threshold:
            used.add(brand)
            results.append({"Source File": src, "Menu Item": line, "Matched Brand": brand,
                            "Transformer Match Score": int(score), "Price": price, "Context": context})
        else:
            results.append({"Source File": src, "Menu Item": line, "Matched Brand": "No Match",
                            "Transformer Match Score": 0, "Price": price, "Context": context})
    df = pd.DataFrame(results)
    return pd.concat([df[df["Matched Brand"] != "No Match"].sort_values(by="Transformer Match Score", ascending=False),
                      df[df["Matched Brand"] == "No Match"]])

if files:
    all_lines = []
    for i, f in enumerate(files):
        show_temperature_bar((i + 1) * int(50 / len(files)))
        lines = [l.strip() for l in extract_lines_from_file(f) if l.strip()]
        st.write(f"📄 OCR / File Lines for {f.name}")
        st.dataframe(pd.DataFrame({"Menu Item": lines}), use_container_width=True)
        all_lines.extend([(line, f.name) for line in lines])

    if brand_file:
        df = pd.read_excel(brand_file) if brand_file.name.endswith("xlsx") else pd.read_csv(brand_file)
        brand_names = df["brand_name"].dropna().astype(str).tolist()
        brand_embeds = st_model.encode(brand_names, convert_to_tensor=True)

        st.write("🧐 LLM Transformer-Based Matching Results:")
        results = []
        for line, src in all_lines:
            results.append(transformer_match([line], brand_names, brand_embeds, threshold, src))

        final_df = pd.concat(results).merge(df, left_on="Matched Brand", right_on="brand_name", how="left")
        total = len(final_df)
        matched = len(final_df[final_df["Matched Brand"] != "No Match"])
        share = int((matched / total) * 100) if total else 0

        st.markdown(f"<h2 style='color: #4CAF50;'>Estimated Menu Share: {share}%</h2>", unsafe_allow_html=True)
        st.dataframe(final_df, use_container_width=True)
        st.download_button("Download Results (CSV)", data=final_df.to_csv(index=False), file_name="SGEYE_Match_Results.csv", mime="text/csv")
