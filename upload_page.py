import streamlit as st
import os
import time

def render_upload_page():
    st.header("Upload PDFs")
    
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully! Redirecting to chat...")
        time.sleep(1)
        st.session_state.page = "chat"
        st.rerun()
