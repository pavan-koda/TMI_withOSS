import streamlit as st
import os
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
import time

def render_upload_page():
    st.header("Admin - Upload and Process PDFs")

    if "engine" not in st.session_state:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            try:
                st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])
            except Exception as e:
                st.error(f"Failed to initialize engine: {e}")
                st.info("Make sure Ollama is running with the Mistral model.")
                return

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploads", uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Processing {uploaded_file.name}... (Creating embeddings)"):
                    try:
                        st.session_state.engine.ingest_pdf(file_path)
                        st.success(f"Successfully processed {uploaded_file.name}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
            else:
                # Check if index exists for existing file
                if not st.session_state.engine.is_pdf_indexed(file_path):
                    with st.spinner(f"Indexing existing file {uploaded_file.name}..."):
                        try:
                            st.session_state.engine.ingest_pdf(file_path)
                            st.success(f"Indexed {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error indexing {uploaded_file.name}: {e}")
                else:
                    st.info(f"{uploaded_file.name} already exists and is indexed.")

    st.subheader("Uploaded PDFs")

    try:
        uploaded_pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    except FileNotFoundError:
        uploaded_pdf_files = []

    if not uploaded_pdf_files:
        st.info("No PDFs uploaded yet.")
    else:
        for pdf_file in uploaded_pdf_files:
            pdf_path = os.path.join("uploads", pdf_file)
            is_indexed = st.session_state.engine.is_pdf_indexed(pdf_path)

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status = "✅" if is_indexed else "⚠️ Not indexed"
                st.write(f"{pdf_file} {status}")
            with col2:
                if st.button("🔄 Rebuild", key=f"rebuild_{pdf_file}", help="Rebuild index with latest settings"):
                    with st.spinner(f"Rebuilding index for {pdf_file}..."):
                        try:
                            st.session_state.engine.rebuild_index(pdf_path)
                            st.success(f"Rebuilt index for {pdf_file}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error rebuilding: {e}")
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{pdf_file}"):
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                    st.session_state.engine.delete_pdf_index(pdf_path)

                    if st.session_state.get("active_pdf") == pdf_file:
                        del st.session_state.active_pdf

                    st.success(f"Deleted {pdf_file}")
                    time.sleep(1)
                    st.rerun()

    # Add rebuild all option
    if uploaded_pdf_files:
        st.markdown("---")
        if st.button("🔄 Rebuild All Indexes", help="Rebuild all PDF indexes with the latest embedding model"):
            progress = st.progress(0)
            for i, pdf_file in enumerate(uploaded_pdf_files):
                pdf_path = os.path.join("uploads", pdf_file)
                st.write(f"Processing {pdf_file}...")
                try:
                    st.session_state.engine.rebuild_index(pdf_path)
                except Exception as e:
                    st.warning(f"Failed to rebuild {pdf_file}: {e}")
                progress.progress((i + 1) / len(uploaded_pdf_files))
            st.success("All indexes rebuilt!")
            time.sleep(1)
            st.rerun()

    if st.button("Go to Chat"):
        st.session_state.page = "chat"
        st.rerun()
