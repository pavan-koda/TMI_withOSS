import streamlit as st
import os

def render_upload_page():
    st.header("Upload PDFs")
    
    # Create the uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the file to the uploads directory
            with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
        
    if st.button("Go to Chat"):
        st.session_state.page = "chat"
        st.experimental_rerun()
