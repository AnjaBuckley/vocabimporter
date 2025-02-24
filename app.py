import streamlit as st
import json
import pandas as pd
from llm_utils import generate_vocabulary
from pathlib import Path
import os
import requests

# Get backend URL from environment variable, default to localhost
BACKEND_URL = os.getenv('WORTWUNDER_BACKEND_URL', 'http://localhost:5000')

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(uploaded_file):
    return json.loads(uploaded_file.getvalue().decode('utf-8'))

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

st.set_page_config(page_title="Vocabulary Generator & Importer", layout="wide")

st.title("Vocabulary Generator & Importer")

tab1, tab2 = st.tabs(["Generate Vocabulary", "Import/Export"])

with tab1:
    st.header("Generate New Vocabulary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input("Topic/Theme", placeholder="e.g., Kitchen utensils, Business vocabulary", key="topic_input")
        language = st.selectbox("Target Language", ["German", "French", "Spanish", "Italian"], key="language_select")
        num_words = st.slider("Number of words to generate", 5, 50, 20, key="num_words_slider")
        
    with col2:
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["A1", "A2", "B1", "B2", "C1", "C2"],
            value="B1",
            key="difficulty_slider"
        )
        include_examples = st.checkbox("Include example sentences", value=True, key="include_examples_checkbox")
    
    if st.button("Generate Vocabulary", key="generate_vocab_btn"):
        with st.spinner("Generating vocabulary..."):
            try:
                vocab_data = generate_vocabulary(
                    topic=topic,
                    language=language,
                    num_words=num_words,
                    difficulty=difficulty,
                    include_examples=include_examples
                )
                
                st.session_state['generated_vocab'] = vocab_data
                
                # Display as table
                df = pd.DataFrame(vocab_data)
                st.dataframe(df, use_container_width=True)
                
                # Export options
                st.write("### Export Options")
                
                # Export to JSON file
                if st.button("Export to JSON", key="export_json_btn"):
                    try:
                        export_filename = st.text_input("Export filename", "vocabulary.json", key="export_filename_input")
                        save_json(st.session_state['generated_vocab'], export_filename)
                        st.success(f"Successfully exported to {export_filename}")
                    except Exception as e:
                        st.error(f"Error exporting file: {str(e)}")
                
                # Export to Wortwunder
                if st.button("Export to Wortwunder", key="export_wortwunder_btn"):
                    success_count, failed_items = export_to_wortwunder(st.session_state['generated_vocab'], topic, difficulty)
                    if success_count > 0:
                        st.success(f"Successfully exported {success_count} words to Wortwunder!")
                    if failed_items:
                        st.error(f"Failed to export: {', '.join(failed_items)}")
                
            except Exception as e:
                st.error(f"Error generating vocabulary: {str(e)}")

with tab2:
    st.header("Import/Export Vocabulary")
    
    # Export section
    if 'generated_vocab' in st.session_state:
        st.subheader("Export Generated Vocabulary")
        export_filename = st.text_input("Export filename", "vocabulary.json", key="export_filename_input_tab2")
        
        if st.button("Export to JSON", key="export_json_btn_tab2"):
            try:
                save_json(st.session_state['generated_vocab'], export_filename)
                st.success(f"Successfully exported to {export_filename}")
            except Exception as e:
                st.error(f"Error exporting file: {str(e)}")
    
    # Import section
    st.subheader("Import Vocabulary")
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            imported_data = load_json(uploaded_file)
            df = pd.DataFrame(imported_data)
            st.dataframe(df, use_container_width=True)
            
            if st.button("Save imported data", key="save_imported_data_btn"):
                st.session_state['generated_vocab'] = imported_data
                st.success("Successfully imported vocabulary!")
                
        except Exception as e:
            st.error(f"Error importing file: {str(e)}")
