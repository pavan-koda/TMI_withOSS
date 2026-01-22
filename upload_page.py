import streamlit as st
import os
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
import time

def render_upload_page():
    st.header("Admin - Upload and Process PDFs")

    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])

    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    process_vision = st.checkbox("Enable Advanced Vision Processing (Slower)", help="Check this if your PDF contains tables, charts, or scanned images. It will take longer to process.")
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploads", uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        st.session_state.engine.ingest_pdf(file_path, use_vision=process_vision)
                        st.success(f"Successfully processed {uploaded_file.name}")
                        time.sleep(1) # Give a moment for the user to see the message
                        st.rerun() # Rerun to update the list of files
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
            else:
                st.info(f"{uploaded_file.name} already exists.")

    st.subheader("Uploaded PDFs")
    
    try:
        uploaded_pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    except FileNotFoundError:
        uploaded_pdf_files = []

    if not uploaded_pdf_files:
        st.info("No PDFs uploaded yet.")
    else:
        for pdf_file in uploaded_pdf_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(pdf_file)
            with col2:
                if st.button(f"Delete", key=f"delete_{pdf_file}"):
                    pdf_path = os.path.join("uploads", pdf_file)
                    
                    # Delete the PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    
                    # Delete the vector index
                    st.session_state.engine.delete_pdf_index(pdf_path)

                    # If the deleted PDF was the active one in chat, clear it
                    if st.session_state.get("active_pdf") == pdf_file:
                        del st.session_state.active_pdf
                    
                    st.success(f"Deleted {pdf_file}")
                    time.sleep(1) # Give a moment for the user to see the message
                    st.rerun()

    if st.button("Go to Chat"):
        st.session_state.page = "chat"
        st.rerun()
