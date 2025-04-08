import streamlit as st
import pandas as pd
import pdfplumber
import easyocr
import ssl
from fuzzywuzzy import fuzz
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import numpy as np
import cv2
import os

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], download_enabled=True)

# Function to extract text from images using EasyOCR with bounding boxes
def extract_text_from_image(image_file):
    image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
   
    if image is None:
        return "", None
   
    # Resize image to reduce memory usage
    image = cv2.resize(image, (1024, 1024))  # Adjust the size as needed
   
    result = reader.readtext(image, detail=1)
    extracted_text = "\n".join([item[1] for item in result])
    return extracted_text, result

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()

# Function to extract menu items from a file
def extract_menu_items(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file), None
    else:
        return extract_text_from_image(file)

# Function to load SGWS brand list
def load_sgws_brands(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl')
    return df

# Function to match menu items to brands and include full row data, also include unmatched items
def match_menu_to_brands(menu_items, brand_df, threshold):
    matched_items = []
    matched_menu_items = set()
   
    for item in menu_items:
        found_match = False
        for _, row in brand_df.iterrows():
            score = fuzz.ratio(item.lower(), row["brand_name"].lower()) if "brand_name" in row else 0
            if score >= threshold:
                matched_items.append(row.tolist() + [item, score])
                matched_menu_items.add(item)
                found_match = True
       
        # Include unmatched items
        if not found_match:
            unmatched_row = [None] * len(brand_df.columns) + [item, "Unmatched"]
            matched_items.append(unmatched_row)
   
    columns = brand_df.columns.tolist() + ["Menu Item", "Score"]
    return pd.DataFrame(matched_items, columns=columns).drop_duplicates()

# Streamlit UI
st.title("Batch Processing: AI Menu Analysis")

st.sidebar.header("Upload Files")
files = st.sidebar.file_uploader("Upload Menus (PDF or Images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
excel_file = st.sidebar.file_uploader("Upload SGWS Brand List (Excel)", type=["xlsx"])
threshold = st.sidebar.slider("Set Matching Threshold", min_value=0, max_value=100, value=51, step=1)

if files and excel_file:
    st.sidebar.success("Files uploaded successfully!")
    sgws_brands_df = load_sgws_brands(excel_file)
   
    all_results = []
   
    for file in files:
        st.subheader(f"Processing: {file.name}")
        extracted_text, _ = extract_menu_items(file)
        menu_items = extracted_text.split("\n") if extracted_text else []
        matched_df = match_menu_to_brands(menu_items, sgws_brands_df, threshold)
        matched_df.insert(0, "File Name", file.name)  # Add file name column
        all_results.append(matched_df)
   
    final_results = pd.concat(all_results, ignore_index=True)
   
    csv_results = final_results.to_csv(index=False).encode('utf-8')
    json_results = final_results.to_json(orient='records', indent=4).encode('utf-8')
   
    st.download_button(label="Download All Results as CSV", data=csv_results, file_name="SGWS_Batch_Matched_Items.csv", mime="text/csv")
    st.download_button(label="Download All Results as JSON", data=json_results, file_name="SGWS_Batch_Matched_Items.json", mime="application/json")
else:
    st.sidebar.warning("Please upload both files to proceed.")