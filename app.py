# app.py

import streamlit as st
import pandas as pd
import requests
import base64
from io import BytesIO

# --- Basic App Configuration ---
st.set_page_config(page_title="Free AI Image Caption Generator", layout="wide")
st.title("🚀 Free AI Image Caption & Metadata Generator")
st.write("100% Free to use! Powered by Hugging Face and Streamlit.")

# --- Helper Function to encode image to base64 ---
def get_image_base64(uploaded_file):
    uploaded_file.seek(0)
    img_bytes = uploaded_file.getvalue()
    encoded_string = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_string

# --- THE CORE: Prompt Generation Function (Simplified for Open Source Models) ---
def generate_prompt(platform, image_type, add_cta, cta_text):
    base_prompt = f"""
Analyze the attached image which is a '{image_type}' file. You are a helpful digital marketing assistant.
First, identify the main subjects, themes, and details of the image.
Then, generate the required metadata for the platform '{platform}'.
Also, suggest a search engine optimized (SEO) filename for the image, using hyphens (e.g., 'a-cute-cat-on-a-sofa.jpg').

"""

    if platform == "Microstock":
        prompt = base_prompt + """
PLATFORM: Microstock

Task 1: Generate two commercial-friendly, keyword-rich titles (under 200 characters).
Task 2: Generate 40 relevant, comma-separated keywords (lowercase, mix of single and double-word). Put the most important keywords first.
Task 3: Suggest one category for Adobe Stock and one for Shutterstock.

OUTPUT FORMAT: Structure your response using these exact headers:

### SEO Filename
[Your filename]

### Title 1
[Your title]

### Title 2
[Your title]

### Keywords
[keywords, here, comma, separated]

### Adobe Stock Category
[Category]

### Shutterstock Categories
[Category]
"""
    elif platform == "Pinterest":
        prompt = base_prompt + """
PLATFORM: Pinterest

Task 1: Generate a catchy Pin Title (max 100 chars).
Task 2: Generate an engaging Pin Description (max 500 chars) with 3-5 relevant hashtags at the end.
Task 3: Suggest a relevant Alt Tag.
Task 4: Suggest a Pinterest Board Name.
"""
        if add_cta:
            prompt += f"IMPORTANT: Include this CTA in the Pin Description: '{cta_text}'\n"
        prompt += """
OUTPUT FORMAT: Structure your response using these exact headers:

### SEO Filename
[Your filename]

### Pin Title
[Your title]

### Pin Description
[Your description with #hashtags]

### Alt Tag
[Your alt tag]

### Suggested Board Name
[Your board name]
"""
    else:  # For Facebook, Instagram, LinkedIn, X
        prompt = base_part + f"""
PLATFORM: {platform}

Task 1: Write an engaging caption for the post.
Task 2: Generate 5-10 relevant hashtags.
"""
        if add_cta:
            prompt += f"IMPORTANT: Include this CTA in your post: '{cta_text}'\n"
        prompt += """
OUTPUT FORMAT: Structure your response using these exact headers:

### SEO Filename
[Your filename]

### Post Caption
[Your caption]

### Hashtags
[#hashtag1, #hashtag2]
"""
    return prompt

# --- Hugging Face API Call Function ---
def get_ai_completion(api_key, b64_image, prompt):
    # This is a popular free Vision-Language model on Hugging Face
    API_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Hugging Face API requires the prompt and image in a specific format
    # The prompt should instruct the model how to behave with the image that follows.
    # The format is "USER: <prompt>\n<image>\nASSISTANT:"
    
    # We are sending the image data directly in the request, not as a URL.
    # So we don't need to format it with special tags in the prompt itself.
    
    try:
        response = requests.post(API_URL, headers=headers, json={
            "inputs": f"USER: {prompt}\nASSISTANT:",
            "parameters": {
                "image": b64_image,
                "max_new_tokens": 1024 # Max length of the generated text
            }
        }, timeout=60) # Increased timeout for slow models

        if response.status_code == 200:
            result = response.json()
            # The actual text is inside 'generated_text' key
            generated_text = result[0].get('generated_text', '')
            # The model's output might include our original prompt, so we clean it
            # We find where ASSISTANT: ends and take the text after it.
            clean_output = generated_text.split("ASSISTANT:")[-1].strip()
            return clean_output
        else:
            return f"Error: Received status code {response.status_code}\nResponse: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred during the API request: {str(e)}"
    except Exception as e:
        return f"An unknown error occurred: {str(e)}"


# --- PARSING FUNCTION ---
def parse_output(text_output):
    # This simple parser assumes the model followed the format.
    try:
        filename = text_output.split("### SEO Filename")[-1].split("###")[0].strip()
        if not filename:
            filename = "seo-friendly-filename.jpg" # Fallback
    except:
        filename = "seo-friendly-filename.jpg" # Fallback
    return filename, text_output

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Generation Settings")
    # Using st.secrets for the API key, as it's the secure way for deployment
    hf_api_key = st.secrets.get("HF_API_TOKEN", "")
    if not hf_api_key:
        st.warning("Please add your Hugging Face API Token to the app's secrets to enable AI generation.")

    platform = st.selectbox(
        "1. Select the target platform:",
        ("Microstock", "Pinterest", "Facebook", "Instagram", "LinkedIn", "X (Twitter)")
    )
    image_type = st.radio(
        "2. Select image type:",
        ("PNG", "JPG", "Vector"), index=0, horizontal=True
    )
    add_cta = st.checkbox("3. Add a Call-to-Action (CTA)?")
    cta_text = ""
    if add_cta:
        cta_text = st.text_input("Enter your CTA text/link", "Learn more on our website!")
    
    st.markdown("---")
    st.info("This app uses a free, open-source AI model. The quality may not be on par with paid services like GPT-4, but it's a powerful free alternative!")

# --- Main Area ---
uploaded_files = st.file_uploader(
    "Upload your images here (multiple files allowed)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)
generate_button = st.button("✨ Generate for Free", type="primary", use_container_width=True)
st.markdown("---")

# Session state to hold results
if 'results' not in st.session_state:
    st.session_state['results'] = []

if generate_button and uploaded_files:
    if not hf_api_key:
        st.error("🚨 Hugging Face API Token is missing! Please add it in the app's secrets.")
    else:
        progress_bar = st.progress(0, text="Starting generation...")
        st.session_state['results'] = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i) / len(uploaded_files), text=f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            with st.spinner(f"AI is analyzing '{uploaded_file.name}'... This can take up to a minute."):
                b64_image = get_image_base64(uploaded_file)
                prompt = generate_prompt(platform, image_type, add_cta, cta_text)
                ai_response = get_ai_completion(hf_api_key, b64_image, prompt)
                
                suggested_filename, output_text = parse_output(ai_response)

                result = {
                    "file_name": uploaded_file.name, "image": uploaded_file.getvalue(),
                    "suggested_filename": suggested_filename, "output": output_text
                }
                st.session_state['results'].append(result)
        
        progress_bar.progress(1.0, text="Generation complete!")
        st.success(f"✅ Successfully generated metadata for {len(uploaded_files)} images!")

# --- Display Results ---
if st.session_state['results']:
    st.header("Generated Results")
    all_outputs_for_csv = []

    for idx, result in enumerate(st.session_state['results']):
        with st.expander(f"🖼️ Results for: **{result['file_name']}**", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result['image'], use_column_width=True)
            with col2:
                st.code(result['output'], language=None) # st.code makes it easy to copy
            
            all_outputs_for_csv.append({
                "original_filename": result['file_name'],
                "suggested_filename": result['suggested_filename'],
                "generated_content": result['output']
            })
    
    if all_outputs_for_csv:
        df = pd.DataFrame(all_outputs_for_csv)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download all results as CSV",
            data=csv,
            file_name=f'{platform.lower()}_metadata_export.csv',
            mime='text/csv',
            use_container_width=True
        )
