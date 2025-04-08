import streamlit as st
import pandas as pd
import pdfplumber
import easyocr
import ssl
import numpy as np
import cv2
import os
from sentence_transformers import SentenceTransformer, util
import re

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="SGEYE: AI Menu Magic", page_icon="📜", layout="wide")
st.markdown("""
    <style>
    body { background-color: #121212; color: #FFFFFF; }
    .stApp { background-color: #121212; }
    .stButton>button { width: 100%; font-size: 18px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: black; border-radius: 8px; }
    h1, h2, h3 { color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

st.video("/Users/jules/Desktop/sgeye/firefly.mp4", start_time=0, loop=True)
#st.video("/Users/Jules.Gerard/Desktop/jules_sgeye.mp4", start_time=0, loop=True)

st.image("sgeye.jpg", width=80)
st.title("📜 SGEYE: AI Menu Magic")

files = st.sidebar.file_uploader("📅 Upload Menus (PDF, Image, CSV, or XLSX)", type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"], accept_multiple_files=True)
brand_file = st.sidebar.file_uploader("📊 Upload Brand List (CSV or XLSX)", type=["csv", "xlsx"])
threshold = st.sidebar.slider("🔎 Transformer Similarity Threshold", 0, 100, 75, 1)

reader = easyocr.Reader(['en'])
progress_placeholder = st.empty()

@st.cache_resource
def load_transformer_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

st_model = load_transformer_model()

def show_temperature_bar(percentage):
    progress_placeholder.progress(percentage / 100.0, text=f"Processing... {percentage}%")

def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text.strip()

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
        df = pd.read_csv(file)
        return df.iloc[:, 0].dropna().tolist()
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        return df.iloc[:, 0].dropna().tolist()
    else:
        return []

def classify_context(line):
    cocktail_keywords = ["shake", "muddle", "stir", "garnish", "recipe", "pour"]
    bottle_keywords = ["bottle", "magnum", "750ml"]
    if any(word in line.lower() for word in cocktail_keywords):
        return "Part of Recipe"
    if any(word in line.lower() for word in bottle_keywords):
        return "Bottle / Standalone"
    return "Standalone Item"

def extract_price(line):
    price_match = re.search(r'\$?\d+\.\d{2}', line)
    return price_match.group(0) if price_match else ""

def transformer_match(menu_lines, brand_names, brand_embeddings, threshold, source_file_name):
    matched_rows = []
    line_embeddings = st_model.encode(menu_lines, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(line_embeddings, brand_embeddings).cpu().numpy()

    used_brands = set()
    for idx, line in enumerate(menu_lines):
        price = extract_price(line)
        row = sim_matrix[idx]
        top_idx = np.argmax(row)
        score = row[top_idx] * 100
        brand = brand_names[top_idx]

        if brand not in used_brands and score >= threshold:
            used_brands.add(brand)
            matched_rows.append({"Source File": source_file_name, "Menu Item": line, "Matched Brand": brand, "Transformer Match Score": int(score), "Price": price, "Context": classify_context(line)})
        else:
            matched_rows.append({"Source File": source_file_name, "Menu Item": line, "Matched Brand": "No Match", "Transformer Match Score": 0, "Price": price, "Context": classify_context(line)})

    df = pd.DataFrame(matched_rows)
    matches = df[df["Matched Brand"] != "No Match"].sort_values(by="Transformer Match Score", ascending=False)
    no_matches = df[df["Matched Brand"] == "No Match"]
    return pd.concat([matches, no_matches])

if files:
    all_lines = []
    for idx, file in enumerate(files):
        show_temperature_bar((idx + 1) * int(50 / len(files)))
        lines = [line.strip() for line in extract_lines_from_file(file) if line.strip()]
        st.write(f"📄 OCR / File Lines for {file.name}")
        st.dataframe(pd.DataFrame({"Menu Item": lines}), use_container_width=True)
        all_lines.extend([(line, file.name) for line in lines])

    if brand_file:
        if brand_file.name.endswith("xlsx"):
            brands_df = pd.read_excel(brand_file)
        else:
            brands_df = pd.read_csv(brand_file)

        # Clean brand names to avoid float tokenization errors
        brand_names = brands_df["brand_name"].dropna().astype(str).tolist()
        brand_embeddings = st_model.encode(brand_names, convert_to_tensor=True)

        st.write("🧐 LLM Transformer-Based Matching Results:")
        results = []
        for line, src_file in all_lines:
            matched_df = transformer_match([line], brand_names, brand_embeddings, threshold, src_file)
            results.append(matched_df)

        final_results = pd.concat(results).merge(brands_df, left_on="Matched Brand", right_on="brand_name", how="left")

        total_items = len(final_results)
        matched_count = len(final_results[final_results["Matched Brand"] != "No Match"])
        menu_share = int((matched_count / total_items) * 100) if total_items > 0 else 0

        st.markdown(f"<h2 style='color: #4CAF50;'>Estimated Menu Share: {menu_share}%</h2>", unsafe_allow_html=True)
        st.dataframe(final_results, use_container_width=True)

        st.download_button("Download Results (CSV)", data=final_results.to_csv(index=False), file_name="SGEYE_Match_Results.csv", mime="text/csv")