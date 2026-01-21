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
from colpali_retriever import ColPaliRetriever
from pdf_processor import PDFProcessor

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
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to {vector_store_path}")

        # 2. Vision Ingestion (Optional - Slower but handles tables/images)
        if use_vision:
            pdf_filename = os.path.basename(pdf_file_path)
            # We use a different naming convention for vision indices
            vision_meta_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}_colpali_meta.pkl")
            
            if not os.path.exists(vision_meta_path):
                logger.info(f"Ingesting PDF (Vision): {pdf_file_path}")
                if not self.pdf_processor:
                    self.pdf_processor = PDFProcessor(output_dir=os.path.join("uploads", "processed"))
                
                # Process PDF to images
                processed_data = self.pdf_processor.process_pdf(pdf_file_path, os.path.join("uploads", "processed", pdf_filename))
                
                if processed_data:
                    if not self.vision_retriever:
                        self.vision_retriever = ColPaliRetriever()
                    
                    # Create Index
                    self.vision_retriever.create_index(
                        page_images=processed_data['page_images'],
                        session_id=pdf_filename,
                        data_dir=VECTOR_STORE_DIR
                    )

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

    def answer_question(self, question, pdf_file_path, callbacks=None, use_vision=False):
        start_time = time.time()
        
        if self._is_greeting(question):
            return {
                "result": "Hello! How can I help you with your document today?",
                "response_time": 0.0,
                "source_documents": []
            }

        result_text = ""
        source_docs = []

        # --- VISION MODE ---
        if use_vision and pdf_file_path != "ALL_PDFS":
            pdf_filename = os.path.basename(pdf_file_path)
            if not self.vision_retriever:
                self.vision_retriever = ColPaliRetriever()
            
            # Search using Vision (Finds the page image that looks like the answer)
            results = self.vision_retriever.search(question, session_id=pdf_filename, data_dir=VECTOR_STORE_DIR, top_k=3)
            
            if results:
                # We found relevant pages visually. Now we need to get the text from those pages 
                # to feed into the LLM (since Mistral is text-based).
                context_text = ""
                
                # Mocking document objects for the UI to display
                class MockDoc:
                    def __init__(self, page, source):
                        self.metadata = {"page": page-1, "source": source}
                        self.page_content = ""

                for res in results:
                    page_num = res['page'] # 1-based
                    context_text += f"\n[Page {page_num} Content]: (Visual Match Score: {res['score']:.2f})\n"
                    source_docs.append(MockDoc(page_num, pdf_filename))

                # Prompt the LLM with the visual context info
                # (In a full VLM setup, we would pass the image, but here we pass the page reference)
                prompt = f"I found these pages visually relevant to the question: {question}\n\nContext:\n{context_text}\n\nPlease answer the question based on the document content found on these pages."
                
                response_text = self.llm.invoke(prompt, config={"callbacks": callbacks})
                result_text = response_text.content if hasattr(response_text, "content") else str(response_text)
            else:
                # If vision fails, fall back to text
                pass

        # --- STANDARD TEXT MODE (Default or Fallback) ---
        if not result_text:
            qa_chain = self._get_qa_chain(pdf_file_path)
            if qa_chain:
                response = qa_chain.invoke({"query": question}, config={"callbacks": callbacks})
                result_text = response["result"]
                source_docs = response.get("source_documents", [])
            else:
                result_text = "It seems this PDF hasn't been processed yet."

        end_time = time.time()
        
        return {
            "result": result_text,
            "response_time": end_time - start_time,
            "source_documents": source_docs
        }
    
    def delete_pdf_index(self, pdf_file_path):
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            del self._vector_store_cache[pdf_filename]
        
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted vector store: {vector_store_path}")
