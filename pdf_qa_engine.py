from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging
import time
import os
import shutil

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_stores"

class PDFQAEngine:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, streaming=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._vector_store_cache = {}
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
        
        # Vision components (lazy loaded)
        self.vision_retriever = None
        self.pdf_processor = None

    def _get_vector_store_path(self, pdf_file_path):
        pdf_filename = os.path.basename(pdf_file_path)
        return os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}.faiss")

    def ingest_pdf(self, pdf_file_path, use_vision=False):
        # 1. Standard Text Ingestion (Always do this as fallback/base)
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        if not os.path.exists(vector_store_path):
            logger.info(f"Ingesting PDF (Text): {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                logger.warning(f"No text found in {pdf_file_path}. Creating placeholder for vector store.")
                chunks = [Document(page_content="[This document contains no extractable text. It may be a scanned image.]", metadata={"source": pdf_file_path, "page": 0})]

            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to {vector_store_path}")

    def _get_qa_chain(self, pdf_file_path):
        vector_store = None

        if pdf_file_path == "ALL_PDFS":
            vector_stores = []
            if os.path.exists(VECTOR_STORE_DIR):
                for filename in os.listdir(VECTOR_STORE_DIR):
                    if filename.endswith(".faiss"):
                        try:
                            vs = FAISS.load_local(os.path.join(VECTOR_STORE_DIR, filename), self.embeddings, allow_dangerous_deserialization=True)
                            vector_stores.append(vs)
                        except Exception as e:
                            logger.error(f"Error loading {filename}: {e}")
            
            if vector_stores:
                vector_store = vector_stores[0]
                for vs in vector_stores[1:]:
                    vector_store.merge_from(vs)
        else:
            vector_store_path = self._get_vector_store_path(pdf_file_path)
            pdf_filename = os.path.basename(pdf_file_path)

            if pdf_filename in self._vector_store_cache:
                vector_store = self._vector_store_cache[pdf_filename]
            elif os.path.exists(vector_store_path):
                vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
                self._vector_store_cache[pdf_filename] = vector_store

        if not vector_store:
            return None

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def _is_greeting(self, text):
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        text = text.lower().strip(" .!,")
        return text in greetings

    def answer_question(self, question, pdf_file_path, callbacks=None, chat_history=None):
        start_time = time.time()
        
        if self._is_greeting(question):
            return {
                "result": "Hello! How can I help you with your document today?",
                "response_time": 0.0,
                "source_documents": []
            }

        qa_chain = self._get_qa_chain(pdf_file_path)
        if not qa_chain:
            return {
                "result": "I cannot find the document index. Please upload the PDF again.",
                "response_time": 0.0,
                "source_documents": []
            }

        response = qa_chain({"query": question}, callbacks=callbacks)
        end_time = time.time()

        return {
            "result": response["result"],
            "response_time": end_time - start_time,
            "source_documents": response["source_documents"]
        }