from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

    def _get_vector_store_path(self, pdf_file_path):
        pdf_filename = os.path.basename(pdf_file_path)
        return os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}.faiss")

    def ingest_pdf(self, pdf_file_path):
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        if os.path.exists(vector_store_path):
            logger.info(f"Vector store for {pdf_file_path} already exists. Skipping ingestion.")
            return

        logger.info(f"Ingesting PDF: {pdf_file_path}")
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store saved to {vector_store_path}")

    def _get_qa_chain(self, pdf_file_path):
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            vector_store = self._vector_store_cache[pdf_filename]
        elif os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            self._vector_store_cache[pdf_filename] = vector_store
        else:
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

    def answer_question(self, question, pdf_file_path, callbacks=None):
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
                "result": "It seems this PDF hasn't been processed yet. Please process it on the upload page.",
                "response_time": 0.0,
                "source_documents": []
            }
        
        response = qa_chain.invoke({"query": question}, config={"callbacks": callbacks})
        end_time = time.time()
        
        return {
            "result": response["result"],
            "response_time": end_time - start_time,
            "source_documents": response.get("source_documents", [])
        }
    
    def delete_pdf_index(self, pdf_file_path):
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            del self._vector_store_cache[pdf_filename]
        
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted vector store: {vector_store_path}")
