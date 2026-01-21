import streamlit as st
import os
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
        self.update_interval = 1

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

def format_duration(seconds):
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.2f} sec"

def render_chat_page():
    st.title("📄 TMI AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(model_name=MODEL_CONFIG["model_name"], base_url=MODEL_CONFIG["base_url"])

    with st.sidebar:
        st.header("Controls")
        if st.button("Go to Admin/Upload Page"):
            st.session_state.page = "upload"
            st.rerun()

        st.markdown("---")
        st.header("Select PDF")
        
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
            
        pdf_files = ["No PDF selected", "Select All"] + [f for f in os.listdir("uploads") if f.endswith(".pdf")]
        selected_pdf = st.selectbox("Choose a PDF to chat with", options=pdf_files)

        if selected_pdf != "No PDF selected":
            if "current_file" not in st.session_state or st.session_state.current_file != selected_pdf:
                st.session_state.current_file = selected_pdf
                st.session_state.messages = []
                st.success(f"Selected {selected_pdf}")
        else:
            st.session_state.current_file = None


        st.markdown("---")
        st.markdown("### Model Config")
        st.info(f"**Model:** {MODEL_CONFIG['model_name']}")
        st.info(f"**Backend:** Ollama")
        
        # Vision Toggle
        use_vision = st.toggle("Enable Vision Mode (ColPali)", value=False, help="Use this for PDFs with tables, charts, or images.")

        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display Metadata
            meta_parts = []
            if "timestamp" in message:
                meta_parts.append(f"🕒 {message['timestamp']}")
            if message["role"] == "assistant" and "response_time" in message:
                meta_parts.append(f"⏱️ {format_duration(message['response_time'])}")
            
            if meta_parts:
                st.caption(" | ".join(meta_parts))
            
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("📚 Reference Pages"):
                    for doc in message["source_documents"]:
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        page = metadata.get("page", "Unknown")
                        if isinstance(page, int): page += 1
                        source = os.path.basename(metadata.get("source", "Unknown"))
                        st.markdown(f"- **Page {page}** ({source})")

    if prompt := st.chat_input("Ask a question about your PDF...", max_chars=2000):
        if not st.session_state.get("current_file"):
            st.warning("Please select a PDF before asking questions.")
            return
        
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {current_time}")

        with st.chat_message("assistant"):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                message_placeholder = st.empty()
            with col2:
                stop_placeholder = st.empty()
            
            if stop_placeholder.button("⏹️", key=f"stop_{current_time}", help="Stop generation"):
                st.stop()

            # Pre-append message to history so it persists if stopped
            st.session_state.messages.append({"role": "assistant", "content": "", "timestamp": current_time})
            current_msg_index = len(st.session_state.messages) - 1
            
            try:
                stream_handler = StreamHandler(message_placeholder, message_context=st.session_state.messages[current_msg_index])
                message_placeholder.markdown("Thinking...")
                
                if st.session_state.current_file == "Select All":
                    pdf_path = "ALL_PDFS"
                else:
                    pdf_path = os.path.join("uploads", st.session_state.current_file)

                response = st.session_state.engine.answer_question(
                    prompt, 
                    pdf_file_path=pdf_path, 
                    callbacks=[stream_handler],
                    use_vision=use_vision
                )
                
                stop_placeholder.empty()
                full_response = response.get("result", "")
                
                # Update final state
                st.session_state.messages[current_msg_index]["content"] = full_response
                st.session_state.messages[current_msg_index]["response_time"] = response.get("response_time", 0)
                st.session_state.messages[current_msg_index]["source_documents"] = response.get("source_documents", [])
                
                message_placeholder.markdown(full_response)
                
                # Show metadata immediately
                st.caption(f"🕒 {current_time} | ⏱️ {format_duration(response['response_time'])}")
                
                if response["source_documents"]:
                    with st.expander("📚 Reference Pages"):
                        for doc in response["source_documents"]:
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            page = metadata.get("page", "Unknown")
                            if isinstance(page, int): page += 1
                            source = os.path.basename(metadata.get("source", "Unknown"))
                            st.markdown(f"- **Page {page}** ({source})")
                            
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
