import os
import json
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from llm_utils import generate_vocabulary, get_llm_provider
import requests

# Load environment variables from .env file if it exists (local development)
# or from Streamlit secrets (cloud deployment)
try:
    env_path = find_dotenv(raise_error_if_not_found=False)
    if env_path:
        print(f"Found .env at: {env_path}")
        load_dotenv(env_path)
except IOError:
    print("No .env file found, using Streamlit secrets")

# Set environment variables from Streamlit secrets if they exist
if 'HUGGINGFACE_API_KEY' in st.secrets:
    os.environ['HUGGINGFACE_API_KEY'] = st.secrets['HUGGINGFACE_API_KEY']
if 'OLLAMA_HOST' in st.secrets:
    os.environ['OLLAMA_HOST'] = st.secrets['OLLAMA_HOST']
if 'WORTWUNDER_BACKEND_URL' in st.secrets:
    os.environ['WORTWUNDER_BACKEND_URL'] = st.secrets['WORTWUNDER_BACKEND_URL']

# Debug: Print environment variables (masked)
hf_key = os.getenv('HUGGINGFACE_API_KEY', '').strip()
if hf_key:
    masked_key = f"{hf_key[:8]}...{hf_key[-4:]}" if len(hf_key) > 12 else "***"
    print(f"Loaded Hugging Face API key: {masked_key}")

# Get backend URL from environment variable, default to localhost
BACKEND_URL = os.getenv('WORTWUNDER_BACKEND_URL', 'http://localhost:5000')

def export_to_wortwunder(vocabulary_items, topic, difficulty):
    """Export vocabulary items to Wortwunder backend"""
    success_count = 0
    failed_items = []
    
    for item in vocabulary_items:
        data = {
            'german_word': item['word'],
            'english_translation': item['translation'],
            'theme': topic,
            'cefr_level': difficulty,
            'word_group_id': None  # Optional, can be added later if needed
        }
        
        try:
            response = requests.post(f"{BACKEND_URL}/api/vocabulary", json=data)
            if response.status_code == 200:
                success_count += 1
            else:
                failed_items.append(item['word'])
        except Exception as e:
            st.error(f"Error exporting {item['word']}: {str(e)}")
            failed_items.append(item['word'])
    
    return success_count, failed_items

st.set_page_config(
    page_title="VocabImporter - Language Learning Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display app header with LLM provider info
def main():
    st.title("📚 VocabImporter")
    st.write("Generate and manage vocabulary for language learning")
    
    # Determine environment
    is_cloud = os.getenv('STREAMLIT_RUNTIME_ENV') is not None or os.getenv('STREAMLIT_RUNTIME') is not None
    
    # Show provider status
    provider = None
    try:
        provider = get_llm_provider()
        provider_name = provider.__class__.__name__.replace('Provider', '')
        st.success(f"✅ Using {provider_name} for vocabulary generation")
    except Exception as e:
        st.error("❌ Error: No LLM provider available")
        error_msg = str(e)
        if "OpenAI error" in error_msg:
            st.warning("Please check your OpenAI API key in Streamlit Cloud settings")
        elif "Ollama error" in error_msg and not is_cloud:
            st.warning("Please ensure Ollama is running locally")
        else:
            st.warning(error_msg)

    tab1, tab2 = st.tabs(["Generate Vocabulary", "Import/Export"])

    with tab1:
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Topic/Theme", placeholder="e.g., Kitchen utensils, Business vocabulary", key="topic_input")
            language = st.selectbox("Target Language", ["German", "French", "Spanish", "Italian"], key="language_select")
            num_words = st.slider("Number of words to generate", 5, 50, 20, key="num_words_slider")
            
        with col2:
            difficulty = st.select_slider(
                "CEFR Level",
                options=["A1", "A2", "B1", "B2", "C1", "C2"],
                value="B1",
                key="difficulty_slider"
            )
            include_examples = st.checkbox("Include example sentences", value=True, key="include_examples_checkbox")
        
        if st.button("Generate Vocabulary", key="generate_vocab_btn"):
            if provider is None:
                st.error("Please fix the LLM provider configuration before generating vocabulary")
                return
            
            if not topic:
                st.warning("Please enter a topic")
                st.stop()
                
            with st.spinner("Generating vocabulary..."):
                try:
                    vocabulary = generate_vocabulary(topic, language, num_words, difficulty, include_examples)
                    st.session_state['generated_vocab'] = vocabulary
                    
                    # Display vocabulary
                    st.write("### Generated Vocabulary")
                    for i, item in enumerate(vocabulary):
                        expander = st.expander(f"{item['word']} - {item['translation']}")
                        with expander:
                            st.write(f"Part of Speech: {item['partOfSpeech']}")
                            st.write(f"Definition: {item['definition']}")
                            if include_examples and 'example' in item:
                                st.write(f"Example: {item['example']}")
                    
                    # Export options
                    st.write("### Export Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Export to JSON file
                        json_str = json.dumps({'words': vocabulary}, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"{language.lower()}_{topic}_vocab.json",
                            mime="application/json",
                            key="download_json_btn"
                        )
                    
                    with col2:
                        # Export to Wortwunder
                        if st.button("🔄 Export to Wortwunder", key="export_wortwunder_btn"):
                            success_count, failed_items = export_to_wortwunder(vocabulary, topic, difficulty)
                            if success_count > 0:
                                st.success(f"Successfully exported {success_count} words to Wortwunder!")
                            if failed_items:
                                st.error(f"Failed to export: {', '.join(failed_items)}")
            
                except Exception as e:
                    st.error(f"Error generating vocabulary: {str(e)}")

    with tab2:
        st.header("Import/Export Vocabulary")
        
        # Import section
        st.subheader("Import from JSON")
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'], key="file_uploader")
        
        if uploaded_file is not None:
            try:
                imported_vocab = json.loads(uploaded_file.getvalue().decode('utf-8'))
                st.session_state['imported_vocab'] = imported_vocab
                st.success("File imported successfully!")
                
                # Display imported vocabulary
                st.write("### Imported Vocabulary")
                for i, item in enumerate(imported_vocab.get('words', [])):
                    expander = st.expander(f"{item['word']} - {item['translation']}")
                    with expander:
                        st.write(f"Part of Speech: {item['partOfSpeech']}")
                        st.write(f"Definition: {item['definition']}")
                        if 'example' in item:
                            st.write(f"Example: {item['example']}")
                
                # Export imported vocabulary to Wortwunder
                if st.button("🔄 Export Imported to Wortwunder", key="export_imported_btn"):
                    if 'imported_vocab' in st.session_state:
                        success_count, failed_items = export_to_wortwunder(
                            st.session_state['imported_vocab']['words'],
                            "Imported",  # Default theme for imported words
                            "B1"  # Default CEFR level for imported words
                        )
                        if success_count > 0:
                            st.success(f"Successfully exported {success_count} imported words to Wortwunder!")
                        if failed_items:
                            st.error(f"Failed to export: {', '.join(failed_items)}")
        
            except Exception as e:
                st.error(f"Error importing file: {str(e)}")

if __name__ == "__main__":
    main()
