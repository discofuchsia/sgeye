import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
from fuzzywuzzy import fuzz
import json
import io

# Load and display the image at 50% width
st.image("sgeye.jpg", caption="A Vision For The Future", width=300)

# Function to extract text from a single PDF without OCR
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()

# Function to extract menu items from a single PDF
def extract_menu_items(pdf_file):
    extracted_text = extract_text_from_pdf(pdf_file)
    return [line.strip() for line in extracted_text.split('\n') if line.strip()]

# Function to filter alcohol-related menu items
def filter_alcohol_menu_items(menu_items):
    alcohol_keywords = ["wine", "beer", "whiskey", "vodka", "rum", "tequila", "gin", "brandy", "liqueur", "bourbon", "scotch", "champagne", "cider"]
    return [item for item in menu_items if any(keyword in item.lower() for keyword in alcohol_keywords)]

# Function to perform fuzzy matching
def match_menu_to_brands(menu_items, brand_list, threshold):
    return sorted([(item, brand, fuzz.ratio(item, brand)) for item in menu_items for brand in brand_list if fuzz.ratio(item, brand) >= threshold], key=lambda x: x[2], reverse=True)

# Load SGWS brand list
def load_sgws_brands(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl')
    return df.iloc[:, 0].dropna().astype(str).str.lower().tolist()

# Streamlit UI
st.title("SGEYE: AI Menu Analysis")

st.sidebar.header("Upload Files")
pdf_files = st.sidebar.file_uploader("Upload Wine Menus (PDF)", type=["pdf"], accept_multiple_files=True)
excel_file = st.sidebar.file_uploader("Upload SGWS Brand List (Excel)", type=["xlsx"])
threshold = st.sidebar.slider("Set Matching Threshold", min_value=0, max_value=100, value=51, step=1)

if pdf_files and excel_file:
    st.sidebar.success("Files uploaded successfully!")
    sgws_brands = load_sgws_brands(excel_file)
   
    for pdf_file in pdf_files:
        st.subheader(f"Processing: {pdf_file.name}")
       
        menu_items = extract_menu_items(pdf_file)
        alcohol_menu_items = filter_alcohol_menu_items(menu_items)
        st.write("**Extracted Alcoholic Menu Items**")
        st.write(alcohol_menu_items)
       
        matched_items = match_menu_to_brands(alcohol_menu_items, sgws_brands, threshold)
        alcohol_match_percentage = min(int((len(matched_items) / len(alcohol_menu_items)) * 100) if alcohol_menu_items else 0, 99)
       
        st.markdown(f"<h1 style='font-size: 24px; font-weight: bold; color: green;'>SGWS Alcohol Menu Share: {alcohol_match_percentage}%</h1>", unsafe_allow_html=True)
       
        results_df = pd.DataFrame(matched_items, columns=["Menu Item", "Matched Brand", "Score"])
        st.dataframe(results_df)
       
        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"Download {pdf_file.name} Matched Items as CSV", data=csv_results, file_name=f"{pdf_file.name}_SGWS_Matched_Items.csv", mime="text/csv")
else:
    st.sidebar.warning("Please upload both files to proceed.")