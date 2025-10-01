# app.py

import streamlit as st
import pandas as pd
import requests

# --- Basic App Configuration ---
st.set_page_config(page_title="Reliable Marketing AI Generator", layout="wide")
st.title("🎯 Reliable Marketing Content AI")
st.header("Turn a simple description of your image into SEO-rich marketing content.")
st.info("✅ **This version is stable.** It no longer depends on unreliable AI vision models.")

# --- API Call for the "Marketing Brain" (Stable Text AI) ---
def generate_marketing_content(api_key: str, prompt: str) -> (bool, str):
    """Calls a powerful and stable text model to generate all marketing content."""
    # This is a flagship, highly reliable instruction-following model.
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 1024}}, timeout=60)
        if response.status_code == 200:
            result = response.json()
            # Clean the output to remove the prompt that the model sometimes repeats
            content = result[0].get('generated_text', '').replace(prompt, '').strip()
            return True, content
        if response.status_code == 503:
            return False, "The AI Marketing model is starting up. This can take up to a minute. Please try again."
        return False, f"AI Marketing Model Error (Code {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"An error occurred with the AI Marketing model: {str(e)}"

# --- Prompt Engineering Function ---
def create_final_prompt(platform: str, user_description: str, add_cta: bool, cta_text: str):
    """Creates the detailed instruction prompt for the text AI based on user input."""
    # This special formatting (`[INST]...[/INST]`) is the correct way to instruct the Mistral model.
    base_instructions = f"""[INST] You are a world-class digital marketing and SEO expert. Your task is to generate compelling metadata for an image.
THE USER DESCRIBES THE IMAGE AS: "{user_description}"

Based on the user's description, follow the instructions for the specified platform below.
---
PLATFORM: {platform}
"""

    if platform == "Microstock":
        prompt = base_instructions + """
Task 1: Generate two highly optimized, SEO-rich titles (under 200 characters each). Make them commercial-friendly and keyword-rich.
Task 2: Generate 49 relevant, comma-separated keywords (all lowercase). Prioritize the most important keywords first. Include a mix of single and double-word keywords.
Task 3: Suggest one ideal category for Adobe Stock and up to two for Shutterstock.

Provide the output EXACTLY in this format, with no extra text or explanations:

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

Provide the output EXACTLY in this format, with no extra text or explanations:

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

Provide the output EXACTLY in this format, with no extra text or explanations:

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

# --- Main App Area ---
st.markdown("### Step 1: Upload Your Image (For Your Reference)")
uploaded_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg", "webp"])

user_description = ""
if uploaded_file:
    st.image(uploaded_file, caption="Your uploaded image", width=300)
    st.markdown("### Step 2: Describe Your Image")
    user_description = st.text_area(
        "Write a simple, one-sentence description of the main subject and theme of your image.",
        placeholder="Example: A 3D illustration of a woman working on a laptop in her cozy small business shop.",
        height=100
    )

st.markdown("### Step 3: Generate Content")
generate_button = st.button("🚀 Generate Marketing Content", type="primary", use_container_width=True, disabled=(not user_description))

if generate_button:
    if not hf_api_key:
        st.error("🚨 Cannot generate! Hugging Face API Token is missing. Please add it in the app's secrets.")
    elif not user_description:
        st.warning("⚠️ Please describe your image in the text box before generating.")
    else:
        with st.spinner("🧠 The AI Marketing Brain is working... This may take a moment."):
            # Create the final prompt for the AI
            final_prompt = create_final_prompt(platform, user_description, add_cta, cta_text)
            
            # Call the AI to get the marketing content
            success, final_content = generate_marketing_content(hf_api_key, final_prompt)
            
            if success:
                st.balloons()
                st.subheader("✅ Generated Content:")
                # Using st.code makes it easy for the user to copy the entire block
                st.code(final_content, language='markdown')

                # Prepare for CSV download
                csv_data = pd.DataFrame([{
                    "user_description": user_description,
                    "platform": platform,
                    "generated_content": final_content
                }]).to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download Result as CSV",
                    data=csv_data,
                    file_name=f'{platform.lower()}_content.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.error(f"A problem occurred. The AI said: {final_content}")
