import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from pdf_qa_engine import PDFQAEngine
from model_config import MODEL_CONFIG
from datetime import datetime

st.set_page_config(page_title="Chat - AI Assistant", page_icon="💬", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", message_context=None):
        self.container = container
        self.text = initial_text
        self.token_count = 0
        self.message_context = message_context
        self.update_interval = 3  # Update UI every N tokens for better performance

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.token_count += 1

        # Update UI less frequently for better performance
        if self.token_count % self.update_interval == 0 or len(token.strip()) == 0:
            self.container.markdown(self.text + "▌")

        # Always update message context for stop functionality
        if self.message_context is not None:
            self.message_context["content"] = self.text

    def on_llm_end(self, _response, **_kwargs) -> None:
        # Final update without cursor
        self.container.markdown(self.text)
        if self.message_context is not None:
            self.message_context["content"] = self.text

def main():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .main {
        background-color: #f8fafc;
    }
    
    /* Style for inline metadata */
    .stCaption {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid #e2e8f0;
    }

    /* Page button styling */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 8px;
        background-color: #fff;
        border: 1px solid #cbd5e1;
        color: #475569;
        transition: all 0.2s ease;
        font-weight: 500;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-color: #94a3b8;
        color: #1e293b;
        background-color: #f8fafc;
    }

    /* Stop button styling - Next to input field */
    div[data-testid="stChatInput"] {
        position: relative;
    }

    /* Hide the default send button when stop button is active */
    body:has(span#stop-btn-anchor) [data-testid="stChatInput"] button {
        display: none !important;
    }

    /* Position the stop button exactly where the send button was */
    div:has(span#stop-btn-anchor) {
        position: fixed !important;
        bottom: 18px !important;
        right: 18px !important;
        z-index: 99999 !important;
    }

    div:has(span#stop-btn-anchor) button {
        width: 36px !important;
        height: 36px !important;
        border-radius: 4px !important;
        background-color: #ef4444 !important;
        color: white !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3) !important;
        transition: all 0.2s ease;
        font-size: 18px !important;
        cursor: pointer !important;
    }

    div:has(span#stop-btn-anchor) button:hover {
        background-color: #dc2626 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
        transform: translateY(-2px);
    }

    div:has(span#stop-btn-anchor) button:active {
        background-color: #b91c1c !important;
        transform: translateY(0);
    }

    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* Chat message animations */
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }
    
    /* Chat container styling */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
    }
    
    .stChatMessage[data-testid="stChatMessage"][data-test-user-name="user"] {
        background-color: #eff6ff;
        border-color: #dbeafe;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Improve chat input responsiveness */
    div[data-testid="stChatInput"] > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        background-color: white;
    }

    div[data-testid="stChatInput"] textarea {
        color: #1e293b !important;
        caret-color: #1e293b;
    }

    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state if accessed directly
    if "engine" not in st.session_state:
        st.session_state.engine = PDFQAEngine(
            model_name=MODEL_CONFIG["model_name"],
            base_url=MODEL_CONFIG["base_url"]
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_file" not in st.session_state:
        st.session_state.active_file = None

    st.title("💬 Chat Assistant")

    if not st.session_state.active_file:
        st.warning("⚠️ No document active. Please go to the Documents page to select a file.")
        if st.button("Go to Documents"):
            st.switch_page("app_pdf_qa.py")
        st.stop()

    # Chat Interface
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            # Metadata display logic...
            if role == "assistant" and "response_time" in message:
                st.caption(f"⏱️ {message['response_time']:.2f}s")

    if prompt := st.chat_input("Ask a question about your PDF..."):
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            stop_placeholder = st.empty()
            
            # Pre-append assistant message
            st.session_state.messages.append({"role": "assistant", "content": "", "timestamp": current_time})
            current_msg_index = len(st.session_state.messages) - 1

            try:
                stream_handler = StreamHandler(message_placeholder, message_context=st.session_state.messages[current_msg_index])
                
                with st.spinner("Thinking..."):
                    response = st.session_state.engine.answer_question(prompt, callbacks=[stream_handler])
                
                st.session_state.messages[current_msg_index]["content"] = response["result"]
                st.session_state.messages[current_msg_index]["response_time"] = response["response_time"]
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()