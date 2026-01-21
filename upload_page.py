import streamlit as st
import os
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG

def render_upload_page():
    st.header("Admin - Upload and Process PDFs")

    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])

    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    st.session_state.engine.ingest_pdf(file_path)
                    st.success(f"Successfully processed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

    if st.button("Go to Chat"):
        st.session_state.page = "chat"
        st.rerun()
