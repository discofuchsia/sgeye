try:
    import streamlit as st
    import pandas as pd
    import pdfplumber
    import re
    import tempfile
    from fuzzywuzzy import fuzz, process
    import json
    import io
except ModuleNotFoundError as e:
    print("Required modules are not installed. Please install them using:")
    print("pip install streamlit pdfplumber pandas openpyxl fuzzywuzzy")
    raise e

# Load and display the image at 50% width
st.image("sgeye.jpg", caption="A Vision For The Future", width=300)

# Function to extract text from PDF without OCR
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_pdf_path = tmp_file.name
   
    with pdfplumber.open(tmp_pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += "\n" + text
   
    return extracted_text.strip()

# Function to extract menu items from PDF, ensuring only brand-like names are extracted
def extract_menu_items(pdf_file):
    extracted_text = extract_text_from_pdf(pdf_file)
    if not extracted_text:
        return []
    menu_items = [re.sub(r'[^a-zA-Z ]', '', line).strip() for line in extracted_text.split('\n')
                  if line.strip() and len(line.split()) > 1]
    return menu_items

# Function to filter alcohol-related menu items with a stricter filter
def filter_alcohol_menu_items(menu_items):
    alcohol_keywords = ["wine", "beer", "whiskey", "vodka", "rum", "tequila", "gin", "brandy", "liqueur", "bourbon", "scotch", "champagne", "cider"]
    return [item for item in menu_items if any(keyword in item.lower().split() for keyword in alcohol_keywords)]

# Function to perform fuzzy matching to brands with adjustable threshold
def match_menu_to_brands(menu_items, brand_list, threshold):
    matches = []
    for item in menu_items:
        for brand in brand_list:
            score = fuzz.ratio(item, brand)
            if score >= threshold:
                matches.append((item, brand, score))
   
    return sorted(matches, key=lambda x: x[2], reverse=True)  # Sort by highest match score

# Load SGWS brand list
def load_sgws_brands(excel_file):
    df = pd.read_excel(excel_file, engine='openpyxl')
    brand_column = df.columns[0]  # Assuming first column has brand names
    sgws_brands = df[brand_column].dropna().astype(str).tolist()
    return [re.sub(r'[^a-zA-Z ]', '', brand).strip().lower() for brand in sgws_brands]

# Streamlit UI
st.title("SGEYE: AI Menu Analysis")

st.sidebar.header("Upload Files")
pdf_file = st.sidebar.file_uploader("Upload Wine Menu (PDF)", type=["pdf"])
excel_file = st.sidebar.file_uploader("Upload SGWS Brand List (Excel)", type=["xlsx"])

# Matching threshold slider
threshold = st.sidebar.slider("Set Matching Threshold", min_value=0, max_value=100, value=51, step=1)

if pdf_file and excel_file:
    st.sidebar.success("Files uploaded successfully!")
    st.write("**AI Menu Reading In Progress...**")
   
    # Extract menu items
    menu_items = extract_menu_items(pdf_file)
   
    # Filter for alcohol-related menu items
    alcohol_menu_items = filter_alcohol_menu_items(menu_items)
   
    st.subheader("Extracted Alcoholic Menu Items")
    st.write(alcohol_menu_items)
   
    # Load SGWS brand list
    sgws_brands = load_sgws_brands(excel_file)
   
    # Match menu to SGWS brand list with adjustable threshold
    matched_items = match_menu_to_brands(alcohol_menu_items, sgws_brands, threshold)
   
    # Calculate percentage of alcohol menu items that match SGWS products
    total_alcohol_items = len(alcohol_menu_items)
    matched_alcohol_items = len(matched_items)
    alcohol_match_percentage = min(int((matched_alcohol_items / total_alcohol_items) * 100) if total_alcohol_items > 0 else 0, 99)  # Ensure it never reaches 100%
   
    st.markdown(f"<h1 style='font-size: 48px; font-weight: bold; color: green;'>SGWS Alcohol Menu Share: {alcohol_match_percentage}%</h1>", unsafe_allow_html=True)
   
    st.subheader("Matching Results")
   
    results_df = pd.DataFrame(matched_items, columns=["Menu Item", "Matched Brand", "Score"])
    st.dataframe(results_df)
   
    # Provide download options
    csv = results_df.to_csv(index=False).encode('utf-8')
    json_data = results_df.to_json(orient='records', indent=4)
   
    st.download_button(label="Download as CSV", data=csv, file_name="SGWS_Matched_Items.csv", mime="text/csv")
    st.download_button(label="Download as JSON", data=json_data, file_name="SGWS_Matched_Items.json", mime="application/json")
else:
    st.sidebar.warning("Please upload both files to proceed.")