import streamlit as st
import os
import tempfile
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
from upload_page import render_upload_page

st.set_page_config(page_title="TMI AI Assistant", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", message_context=None):
        self.container = container
        self.text = initial_text
        self.token_count = 0
        self.message_context = message_context
        self.update_interval = 3

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.token_count += 1
        if self.token_count % self.update_interval == 0 or len(token.strip()) == 0:
            self.container.markdown(self.text + "▌")
        if self.message_context is not None:
            self.message_context["content"] = self.text

    def on_llm_end(self, _response, **_kwargs) -> None:
        self.container.markdown(self.text)
        if self.message_context is not None:
            self.message_context["content"] = self.text

def render_chat_page():
    st.title("📄 TMI AI Assistant")

    st.markdown("""
    <style>
    /* ... existing styles ... */
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    with st.sidebar:
        st.header("Controls")
        if st.button("Go to Upload Page"):
            st.session_state.page = "upload"
            st.experimental_rerun()

        st.markdown("---")
        st.header("Select PDF")

        # Create uploads directory if it doesn't exist
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
                st.session_state.is_generating = False
                st.session_state.stop_requested = False
                st.experimental_rerun()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            _, col_msg = st.columns([2, 10])
            with col_msg:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"🕒 {message['timestamp']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                meta_text = ""
                if "timestamp" in message:
                    meta_text += f"🕒 {message['timestamp']}"
                if "response_time" in message and message["response_time"] > 0:
                    rt = message["response_time"]
                    meta_text += f" | ⏱️ {rt:.2f}s"
                if "pages" in message and message["pages"]:
                    st.caption(meta_text + " | 📄 Pages: " + ", ".join(map(str, message["pages"])))
                elif meta_text:
                    st.caption(meta_text)

    if prompt := st.chat_input("Ask a question about your PDF..."):
        if "current_file" not in st.session_state or st.session_state.current_file == "No PDF selected":
            st.warning("Please select a PDF before asking questions.")
            return
            
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        st.session_state.messages.append({"role": "assistant", "content": "", "timestamp": datetime.now().strftime("%H:%M:%S"), "response_time": 0, "pages": []})
        current_msg_index = len(st.session_state.messages) - 1

        st.experimental_rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == "":
        message = st.session_state.messages[-2]
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
            st.caption(f"🕒 {message['timestamp']}")

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            try:
                stream_handler = StreamHandler(message_placeholder, message_context=st.session_state.messages[-1])
                with st.spinner("🔍 Retrieving relevant context..."):
                    response = st.session_state.engine.answer_question(st.session_state.messages[-2]["content"], callbacks=[stream_handler])
                
                answer_text = response["result"]
                time_taken = response["response_time"]
                source_docs = response["source_documents"]
                pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))) if source_docs else []

                st.session_state.messages[-1]["content"] = answer_text
                st.session_state.messages[-1]["response_time"] = time_taken
                st.session_state.messages[-1]["pages"] = pages
                
                message_placeholder.markdown(answer_text)
                st.experimental_rerun()

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")


def main():
    if "page" not in st.session_state:
        st.session_state.page = "chat"

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "upload":
        render_upload_page()

if __name__ == "__main__":
    main()
