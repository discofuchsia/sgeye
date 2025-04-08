import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
from fuzzywuzzy import fuzz
import json
import io
import easyocr
import ssl
import certifi

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], download_enabled=True)

st.image("sgeye.jpg", caption="SGEYE - AI Menu Analysis", use_container_width=True)

# Function to extract text from images using EasyOCR
def extract_text_from_image(image_file):
    image_bytes = image_file.read()
    result = reader.readtext(image_bytes, detail=0)
    return "\n".join(result)

# Function to extract text from a single PDF without OCR
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()

# Function to extract menu items from a single PDF or image
def extract_menu_items(file):
    if file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(file)
    else:
        extracted_text = extract_text_from_image(file)
   
    return [line.strip() for line in extracted_text.split('\n') if line.strip()]

# Function to filter alcohol-related menu items with a more relaxed approach
def filter_alcohol_menu_items(menu_items):
    alcohol_keywords = ["wine", "beer", "whiskey", "vodka", "rum", "tequila", "gin", "brandy", "liqueur", "bourbon", "scotch", "champagne", "cider", "ale", "cocktail", "martini", "mezcal", "sake"]
    return [item for item in menu_items if any(keyword.lower() in item.lower() for keyword in alcohol_keywords) or len(item.split()) <= 4]

# Function to perform fuzzy matching (case insensitive)
def match_menu_to_brands(menu_items, brand_list, threshold):
    return sorted([(item, brand, fuzz.ratio(item.lower(), brand.lower())) for item in menu_items for brand in brand_list if fuzz.ratio(item.lower(), brand.lower()) >= threshold], key=lambda x: x[2], reverse=True)

# Load SGWS brand list
def load_sgws_brands(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl')
    brand_column = "brand_name" if "brand_name" in df.columns else df.columns[0]
    return df[brand_column].dropna().astype(str).str.lower().tolist()

# Streamlit UI
st.title("SGEYE: AI Menu Analysis")

st.sidebar.header("Upload Files")
files = st.sidebar.file_uploader("Upload Menus (PDF or Images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
excel_file = st.sidebar.file_uploader("Upload SGWS Brand List (Excel)", type=["xlsx"])
threshold = st.sidebar.slider("Set Matching Threshold", min_value=0, max_value=100, value=68, step=1)

if files and excel_file:
    st.sidebar.success("Files uploaded successfully!")
    sgws_brands = load_sgws_brands(excel_file)
   
    for idx, file in enumerate(files):
        st.subheader(f"Processing: {file.name}")
       
        menu_items = extract_menu_items(file)
        alcohol_menu_items = filter_alcohol_menu_items(menu_items)
        st.write("**Extracted Alcoholic Menu Items**")
        st.write(alcohol_menu_items)
       
        matched_items = match_menu_to_brands(alcohol_menu_items, sgws_brands, threshold)
        alcohol_match_percentage = min(int((len(matched_items) / len(alcohol_menu_items)) * 100) if alcohol_menu_items else 0, 99)
       
        st.markdown(f"<h1 style='font-size: 32px; font-weight: bold; color: green;'>SGWS Menu Share: {alcohol_match_percentage}%</h1>", unsafe_allow_html=True)
       
        results_df = pd.DataFrame(matched_items, columns=["Menu Item", "Matched Brand", "Score"])
        st.dataframe(results_df)
       
        csv_results = results_df.to_csv(index=False).encode('utf-8')
        json_results = results_df.to_json(orient='records', indent=4).encode('utf-8')
       
        st.download_button(label=f"Download {file.name} Matched Items as CSV", data=csv_results, file_name=f"{file.name}_SGWS_Matched_Items.csv", mime="text/csv", key=f"csv_{idx}")
        st.download_button(label=f"Download {file.name} Matched Items as JSON", data=json_results, file_name=f"{file.name}_SGWS_Matched_Items.json", mime="application/json", key=f"json_{idx}")
else:
    st.sidebar.warning("Please upload both files to proceed.")