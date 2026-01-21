import streamlit as st
import os
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
from upload_page import render_upload_page

st.set_page_config(page_title="TMI AI Assistant", layout="wide")

class StreamHandler(BaseCallbackHandler):
    # ... (StreamHandler class remains the same) ...

def render_chat_page():
    st.title("📄 TMI AI Assistant")

    # ... (CSS styles remain the same) ...

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])

    with st.sidebar:
        st.header("Select PDF")
        
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
            
        pdf_files = ["No PDF selected"] + [f for f in os.listdir("uploads") if f.endswith(".pdf")]
        selected_pdf = st.selectbox("Choose a PDF to chat with", options=pdf_files)

        if selected_pdf != "No PDF selected":
            if "current_file" not in st.session_state or st.session_state.current_file != selected_pdf:
                with st.spinner(f"Processing {selected_pdf}..."):
                    pdf_path = os.path.join("uploads", selected_pdf)
                    try:
                        st.session_state.engine.ingest_pdf(pdf_path)
                        st.session_state.current_file = selected_pdf
                        st.session_state.messages = []
                        st.success("PDF Processed Successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

        st.markdown("---")
        st.markdown("### Model Config")
        st.info(f"**Model:** {MODEL_CONFIG['model_name']}")
        st.info(f"**Backend:** Ollama")

        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # ... (Chat message display loop remains the same) ...

    if prompt := st.chat_input("Ask a question about your PDF..."):
        if "current_file" not in st.session_state or st.session_state.current_file == "No PDF selected":
            st.warning("Please select a PDF before asking questions.")
            return
            
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        st.session_state.messages.append({"role": "assistant", "content": "", "timestamp": datetime.now().strftime("%H:%M:%S"), "response_time": 0, "pages": []})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == "":
        # ... (Response generation logic remains the same, but with st.rerun()) ...
        try:
            # ...
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")


def main():
    if "page" not in st.session_state:
        # If uploads dir is empty or doesn't exist, go to upload page
        if not os.path.exists("uploads") or not any(f.endswith(".pdf") for f in os.listdir("uploads")):
            st.session_state.page = "upload"
        else:
            st.session_state.page = "chat"

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "upload":
        render_upload_page()

if __name__ == "__main__":
    main()
