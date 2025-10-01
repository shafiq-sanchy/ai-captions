# app.py

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import time

# --- Basic App Configuration ---
st.set_page_config(page_title="Inbound Marketing AI Generator", layout="wide")
st.title("🎯 Inbound Marketing AI Generator")
st.header("Generate SEO-rich titles, keywords & captions that drive traffic.")
st.markdown("Powered by a 2-step AI process: an AI 'Vision' model describes the image, then an AI 'Marketing' model writes the content.")

# --- API Call for AI #1 (The "Eyes" - Image to Text) ---
def get_image_description(api_key: str, image_bytes: bytes) -> (bool, str):
    """Calls a stable image-to-text model to get a factual description."""
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=45)
        if response.status_code == 200:
            result = response.json()
            description = result[0].get('generated_text', '')
            return True, description
        # Handle model loading state
        if response.status_code == 503:
            return False, "The AI 'Vision' model is starting up. This can take up to a minute. Please try again."
        return False, f"AI 'Vision' Error (Code {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"An error occurred with the AI 'Vision' model: {str(e)}"

# --- API Call for AI #2 (The "Brain" - Text Generation) ---
def generate_metadata_from_text(api_key: str, prompt: str) -> (bool, str):
    """Calls a powerful text model to generate marketing content based on a description."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 1024}}, timeout=60)
        if response.status_code == 200:
            result = response.json()
            content = result[0].get('generated_text', '').replace(prompt, '').strip() # Clean the output
            return True, content
        if response.status_code == 503:
             return False, "The AI 'Marketing' model is starting up. This can take up to a minute. Please try again."
        return False, f"AI 'Marketing' Error (Code {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"An error occurred with the AI 'Marketing' model: {str(e)}"

# --- Prompt Engineering Function ---
def create_final_prompt(platform: str, image_description: str, add_cta: bool, cta_text: str):
    """Creates the detailed instruction prompt for the text AI."""
    base_instructions = f"""[INST] You are a world-class digital marketing and SEO expert. Your task is to generate compelling metadata for an image.
THE IMAGE IS DESCRIBED AS: "{image_description}"

Based on this description, follow the instructions for the specified platform below.
---
PLATFORM: {platform}
"""

    if platform == "Microstock":
        prompt = base_instructions + """
Task 1: Generate two highly optimized, SEO-rich titles (under 200 characters each). Make them commercial-friendly and keyword-rich.
Task 2: Generate 49 relevant, comma-separated keywords (all lowercase). Prioritize the most important keywords first. Include a mix of single and double-word keywords.
Task 3: Suggest one ideal category for Adobe Stock and up to two for Shutterstock.

Provide the output EXACTLY in this format, with no extra text:

### SEO Filename
[Generate an SEO-friendly filename using hyphens]

### Title 1
[Title 1]

### Title 2
[Title 2]

### Keywords
[49 keywords, comma, separated]

### Adobe Stock Category
[Category]

### Shutterstock Categories
[Category 1, Category 2]
[/INST]"""

    elif platform == "Pinterest":
        cta_instruction = f"IMPORTANT: Naturally integrate this Call-to-Action in the description: '{cta_text}'" if add_cta else ""
        prompt = base_instructions + f"""
Task 1: Generate a catchy, SEO-friendly Pin Title (max 100 chars).
Task 2: Generate an engaging Pin Description (max 500 chars) with 3-5 relevant hashtags at the end. {cta_instruction}
Task 3: Suggest a relevant, descriptive Alt Tag.
Task 4: Suggest a relevant Pinterest Board Name.

Provide the output EXACTLY in this format, with no extra text:

### SEO Filename
[Generate an SEO-friendly filename]

### Pin Title
[Title]

### Pin Description
[Description with #hashtags]

### Alt Tag
[Alt tag]

### Suggested Board Name
[Board name]
[/INST]"""
    else: # Facebook, Instagram, LinkedIn, X
        cta_instruction = f"IMPORTANT: Seamlessly include this Call-to-Action in the post: '{cta_text}'" if add_cta else ""
        prompt = base_instructions + f"""
Task 1: Write an engaging and high-converting caption for the post. For Facebook/LinkedIn, make it more detailed. For Instagram/X, keep it concise and impactful.
Task 2: Generate 5-10 highly relevant hashtags.

{cta_instruction}

Provide the output EXACTLY in this format, with no extra text:

### SEO Filename
[Generate an SEO-friendly filename]

### Post Caption
[Caption]

### Hashtags
[#hashtag1, #hashtag2]
[/INST]"""

    return prompt

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Generation Settings")
    hf_api_key = st.secrets.get("HF_API_TOKEN", "")
    if not hf_api_key:
        st.warning("Please add your Hugging Face API Token to the app's secrets.")
    else:
        st.success("Hugging Face API Token loaded!")

    platform = st.selectbox("1. Select Target Platform:", ("Microstock", "Pinterest", "Facebook", "Instagram", "LinkedIn", "X (Twitter)"))
    add_cta = st.checkbox("2. Add a Call-to-Action (CTA)?")
    cta_text = ""
    if add_cta:
        cta_text = st.text_input("Enter your CTA text/link", "Visit our website for more!")

# --- Main Area ---
uploaded_files = st.file_uploader("Upload your images here", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
generate_button = st.button("🚀 Generate Marketing Content", type="primary", use_container_width=True)

if 'results' not in st.session_state:
    st.session_state['results'] = []

if generate_button and uploaded_files:
    if not hf_api_key:
        st.error("🚨 Cannot generate! Hugging Face API Token is missing. Please add it in the app's secrets.")
    else:
        st.session_state['results'] = []
        progress_bar = st.progress(0, text="Starting...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            # --- AI Step 1: Get Image Description ---
            st.write(f"Analyzing image {i+1}/{len(uploaded_files)}: `{uploaded_file.name}`...")
            image_bytes = uploaded_file.getvalue()
            success, description = get_image_description(hf_api_key, image_bytes)

            if not success:
                st.error(f"Could not analyze `{uploaded_file.name}`. Reason: {description}")
                continue
            
            # --- AI Step 2: Generate Marketing Content ---
            st.write(f"Generating marketing content for `{uploaded_file.name}`...")
            final_prompt = create_final_prompt(platform, description, add_cta, cta_text)
            success, final_content = generate_metadata_from_text(hf_api_key, final_prompt)
            
            if not success:
                st.error(f"Could not generate content for `{uploaded_file.name}`. Reason: {final_content}")
                continue

            result = {"file_name": uploaded_file.name, "image_bytes": image_bytes, "content": final_content}
            st.session_state['results'].append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success("✅ Content generation complete!")

# --- Display Results ---
if st.session_state['results']:
    st.markdown("---")
    st.header("Generated Results")
    all_outputs_for_csv = []
    
    for idx, result in enumerate(st.session_state['results']):
        with st.expander(f"🖼️ Results for: **{result['file_name']}**", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result['image_bytes'], use_column_width=True)
            with col2:
                st.code(result['content'], language='markdown') # Using st.code for easy copying
            
            all_outputs_for_csv.append({
                "original_filename": result['file_name'],
                "generated_content": result['content']
            })
    
    if all_outputs_for_csv:
        df = pd.DataFrame(all_outputs_for_csv)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download all results as CSV",
            data=csv_data,
            file_name=f'{platform.lower()}_marketing_content.csv',
            mime='text/csv',
            use_container_width=True
        )
