from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import logging
import time
import os
import re
import shutil
from colpali_retriever import ColPaliRetriever
from pdf_processor import PDFProcessor
import fitz
from config import EMBEDDING_CONFIG, RERANKER_CONFIG, RETRIEVAL_CONFIG, PDF_CONFIG

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_stores"


class PDFQAEngine:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, streaming=True)

        # Use better embedding model from config
        embedding_model = EMBEDDING_CONFIG.get('model_name', 'BAAI/bge-base-en-v1.5')
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Important for BGE models
        )

        # Initialize cross-encoder reranker for improved accuracy
        self.reranker = None
        if RERANKER_CONFIG.get('enabled', True):
            try:
                reranker_model = RERANKER_CONFIG.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info(f"Loading reranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")

        self._vector_store_cache = {}
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)

        # Vision components (lazy loaded)
        self.vision_retriever = None
        self.pdf_processor = None

    def _get_vector_store_path(self, pdf_file_path):
        pdf_filename = os.path.basename(pdf_file_path)
        return os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}.faiss")

    def _preprocess_text(self, text):
        """Clean and normalize text for better chunking and retrieval."""
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove page headers/footers patterns (common in PDFs)
        text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _enhance_chunks_with_context(self, chunks, pdf_file_path):
        """Add contextual information to chunks for better retrieval."""
        enhanced_chunks = []
        pdf_name = os.path.basename(pdf_file_path)

        for i, chunk in enumerate(chunks):
            # Add page context to metadata
            page_num = chunk.metadata.get('page', 0)

            # Create enhanced content with context header
            enhanced_content = f"[Document: {pdf_name}, Page: {page_num + 1}]\n{chunk.page_content}"

            enhanced_chunks.append(Document(
                page_content=enhanced_content,
                metadata={
                    **chunk.metadata,
                    'chunk_index': i,
                    'original_content': chunk.page_content
                }
            ))

        return enhanced_chunks

    def ingest_pdf(self, pdf_file_path, use_vision=False):
        """Ingest PDF with improved chunking strategy."""
        # 1. Standard Text Ingestion (Always do this as fallback/base)
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        if not os.path.exists(vector_store_path):
            logger.info(f"Ingesting PDF (Text): {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()

            # Preprocess text in each document
            for doc in documents:
                doc.page_content = self._preprocess_text(doc.page_content)

            # Use improved chunking with paragraph-aware separators
            chunk_size = PDF_CONFIG.get('chunk_size', 800)
            chunk_overlap = PDF_CONFIG.get('chunk_overlap', 200)
            separators = PDF_CONFIG.get('separators', ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "])

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False
            )
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                logger.warning(f"No text found in {pdf_file_path}. Creating placeholder for vector store.")
                chunks = [Document(
                    page_content="[This document contains no extractable text. It may be a scanned image.]",
                    metadata={"source": pdf_file_path, "page": 0}
                )]

            # Enhance chunks with contextual information
            chunks = self._enhance_chunks_with_context(chunks, pdf_file_path)

            logger.info(f"Created {len(chunks)} chunks from PDF")

            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to {vector_store_path}")

        # 2. Vision Ingestion (Optional - Slower but handles tables/images)
        if use_vision:
            pdf_filename = os.path.basename(pdf_file_path)
            vision_meta_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}_colpali_meta.pkl")

            if not os.path.exists(vision_meta_path):
                logger.info(f"Ingesting PDF (Vision Mode - ColPali): {pdf_file_path}")
                if not self.pdf_processor:
                    self.pdf_processor = PDFProcessor()

                # Process PDF to images
                processed_data = self.pdf_processor.process_pdf(
                    pdf_file_path,
                    os.path.join("uploads", "processed", pdf_filename)
                )

                if processed_data:
                    if not self.vision_retriever:
                        self.vision_retriever = ColPaliRetriever()

                    # Create Index
                    self.vision_retriever.create_index(
                        page_images=processed_data['page_images'],
                        session_id=pdf_filename,
                        data_dir=VECTOR_STORE_DIR
                    )

    def _rerank_documents(self, query, documents, top_k=5):
        """Rerank documents using cross-encoder for better relevance."""
        if not self.reranker or not documents:
            return documents[:top_k]

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]

            # Get relevance scores
            scores = self.reranker.predict(pairs)

            # Sort documents by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Log reranking results for debugging
            logger.info(f"Reranking results - Top scores: {[f'{s:.3f}' for _, s in scored_docs[:3]]}")

            # Return top_k documents
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return documents[:top_k]

    def _preprocess_query(self, query):
        """Preprocess and potentially expand the query for better matching."""
        # Clean the query
        query = query.strip()

        # Handle common question patterns for better semantic matching
        # Remove filler words that don't add semantic meaning
        query = re.sub(r'\b(please|kindly|can you|could you|tell me|what is|what are)\b', '', query, flags=re.IGNORECASE)
        query = query.strip()

        # If query is very short, it might be a keyword search - keep it as is
        if len(query.split()) <= 2:
            return query

        return query

    def _get_qa_chain(self, pdf_file_path):
        """Create QA chain with improved retrieval settings."""
        vector_store = None

        if pdf_file_path == "ALL_PDFS":
            vector_stores = []
            if os.path.exists(VECTOR_STORE_DIR):
                for filename in os.listdir(VECTOR_STORE_DIR):
                    if filename.endswith(".faiss"):
                        try:
                            vs = FAISS.load_local(
                                os.path.join(VECTOR_STORE_DIR, filename),
                                self.embeddings,
                                allow_dangerous_deserialization=True
                            )
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
                vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._vector_store_cache[pdf_filename] = vector_store

        if not vector_store:
            return None

        # Get retrieval settings from config
        initial_k = RETRIEVAL_CONFIG.get('initial_k', 10)
        use_mmr = RETRIEVAL_CONFIG.get('use_mmr', True)
        mmr_lambda = RETRIEVAL_CONFIG.get('mmr_lambda', 0.7)

        # Use MMR (Maximum Marginal Relevance) for diverse, relevant results
        if use_mmr:
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": initial_k,
                    "lambda_mult": mmr_lambda,  # Balance between relevance and diversity
                    "fetch_k": initial_k * 2  # Fetch more candidates for MMR selection
                }
            )
        else:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": initial_k}
            )

        # Improved prompt template for more accurate answers
        prompt_template = """You are a precise document analysis assistant. Your task is to answer questions ONLY using the information provided in the context below.

CRITICAL INSTRUCTIONS:
1. Read the context carefully and thoroughly before answering.
2. Base your answer STRICTLY on the information in the context - do not add external knowledge.
3. If the exact answer is in the context, quote or paraphrase it accurately.
4. If the context contains partial information, provide what is available and note what's missing.
5. If the answer is NOT in the context, respond: "I cannot find this information in the provided document."
6. For numerical data, dates, or specific facts - be precise and cite the source page if mentioned.
7. If multiple pieces of context are relevant, synthesize them into a coherent answer.

Context from the document:
{context}

Question: {question}

Provide a clear, accurate answer based solely on the context above:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
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

    def _extract_relevant_context(self, documents, query):
        """Extract and format the most relevant parts of retrieved documents."""
        context_parts = []
        seen_content = set()

        for doc in documents:
            # Get original content if available, otherwise use page_content
            content = doc.metadata.get('original_content', doc.page_content)

            # Avoid duplicate content
            content_hash = hash(content[:100])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            # Add page reference
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(f"[Page {page + 1 if isinstance(page, int) else page}]: {content}")

        return "\n\n---\n\n".join(context_parts)

    def answer_question(self, question, pdf_file_path, callbacks=None, use_vision=False, chat_history=None):
        """Answer question with improved retrieval, reranking, and context handling."""
        start_time = time.time()

        if self._is_greeting(question):
            return {
                "result": "Hello! How can I help you with your document today?",
                "response_time": 0.0,
                "source_documents": []
            }

        if chat_history is None:
            chat_history = []

        # Format chat history for context
        history_text = ""
        for msg in chat_history[-4:]:  # Only use last 4 messages for context
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if role and content:
                history_text += f"{role}: {content}\n"

        # Preprocess the query for better matching
        processed_query = self._preprocess_query(question)

        result_text = ""
        source_docs = []

        # --- VISION MODE ---
        if use_vision and pdf_file_path != "ALL_PDFS":
            pdf_filename = os.path.basename(pdf_file_path)

            results = []
            vision_meta_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}_colpali_meta.pkl")

            if os.path.exists(vision_meta_path):
                if not self.vision_retriever:
                    self.vision_retriever = ColPaliRetriever()
                results = self.vision_retriever.search(
                    question,
                    session_id=pdf_filename,
                    data_dir=VECTOR_STORE_DIR,
                    top_k=3
                )
            else:
                logger.warning(f"Vision index not found for {pdf_filename}. Falling back to text mode.")

            if results:
                logger.info(f"Vision Mode: Found {len(results)} relevant pages using ColPali.")
                context_text = ""

                class MockDoc:
                    def __init__(self, page, source, content=""):
                        self.metadata = {"page": page-1, "source": source}
                        self.page_content = content

                try:
                    doc = fitz.open(pdf_file_path)
                except Exception as e:
                    logger.error(f"Error opening PDF for vision context: {e}")
                    doc = None

                for res in results:
                    page_num = int(res['page'])
                    page_content = ""
                    try:
                        if doc and 0 <= page_num-1 < len(doc):
                            page_content = doc[page_num-1].get_text()
                            page_content = self._preprocess_text(page_content)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")

                    if not page_content or not page_content.strip():
                        page_content = "[Visual Content Only - This page contains images/tables that cannot be extracted as text.]"

                    context_text += f"\n[Page {page_num} - Visual Match Score: {res['score']:.2f}]:\n{page_content}\n"
                    source_docs.append(MockDoc(page_num, pdf_filename, page_content))

                if doc:
                    doc.close()

                # Hybrid: Also include text-based retrieval results
                vector_store_path = self._get_vector_store_path(pdf_file_path)
                vector_store = None

                if pdf_filename in self._vector_store_cache:
                    vector_store = self._vector_store_cache[pdf_filename]
                elif os.path.exists(vector_store_path):
                    try:
                        vector_store = FAISS.load_local(
                            vector_store_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        self._vector_store_cache[pdf_filename] = vector_store
                    except Exception as e:
                        logger.error(f"Error loading vector store for hybrid search: {e}")

                if vector_store:
                    try:
                        # Get more text results for hybrid mode
                        text_docs = vector_store.similarity_search(processed_query, k=4)

                        # Rerank text results
                        if self.reranker and text_docs:
                            text_docs = self._rerank_documents(question, text_docs, top_k=3)

                        if text_docs:
                            context_text += "\n\n--- Additional Text Context ---\n"
                            for tdoc in text_docs:
                                content = tdoc.metadata.get('original_content', tdoc.page_content)
                                context_text += f"\n{content}\n"
                                source_docs.append(tdoc)
                    except Exception as e:
                        logger.warning(f"Text retrieval failed in vision mode: {e}")

                # Generate answer with vision context
                prompt = f"""Based on the following document content, answer the question accurately.

Previous conversation:
{history_text}

Document Content:
{context_text}

Question: {question}

Provide a clear, accurate answer based on the document content:"""

                response_text = self.llm.invoke(prompt, config={"callbacks": callbacks})
                result_text = response_text.content if hasattr(response_text, "content") else str(response_text)

        # --- STANDARD TEXT MODE (Default or Fallback) ---
        if not result_text:
            # First, get documents directly for reranking
            vector_store_path = self._get_vector_store_path(pdf_file_path)
            pdf_filename = os.path.basename(pdf_file_path) if pdf_file_path != "ALL_PDFS" else "ALL_PDFS"
            vector_store = None

            if pdf_file_path == "ALL_PDFS":
                # Load all vector stores
                vector_stores = []
                if os.path.exists(VECTOR_STORE_DIR):
                    for filename in os.listdir(VECTOR_STORE_DIR):
                        if filename.endswith(".faiss"):
                            try:
                                vs = FAISS.load_local(
                                    os.path.join(VECTOR_STORE_DIR, filename),
                                    self.embeddings,
                                    allow_dangerous_deserialization=True
                                )
                                vector_stores.append(vs)
                            except Exception as e:
                                logger.error(f"Error loading {filename}: {e}")

                if vector_stores:
                    vector_store = vector_stores[0]
                    for vs in vector_stores[1:]:
                        vector_store.merge_from(vs)
            else:
                if pdf_filename in self._vector_store_cache:
                    vector_store = self._vector_store_cache[pdf_filename]
                elif os.path.exists(vector_store_path):
                    vector_store = FAISS.load_local(
                        vector_store_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self._vector_store_cache[pdf_filename] = vector_store

            if vector_store:
                logger.info(f"Standard Mode: Using enhanced Text RAG with reranking.")

                # Get retrieval settings
                initial_k = RETRIEVAL_CONFIG.get('initial_k', 10)
                use_mmr = RETRIEVAL_CONFIG.get('use_mmr', True)
                mmr_lambda = RETRIEVAL_CONFIG.get('mmr_lambda', 0.7)
                reranker_top_k = RERANKER_CONFIG.get('top_k', 5)

                # Retrieve initial documents using MMR
                if use_mmr:
                    retrieved_docs = vector_store.max_marginal_relevance_search(
                        processed_query,
                        k=initial_k,
                        lambda_mult=mmr_lambda,
                        fetch_k=initial_k * 2
                    )
                else:
                    retrieved_docs = vector_store.similarity_search(processed_query, k=initial_k)

                # Apply reranking for better relevance
                if self.reranker and retrieved_docs:
                    logger.info(f"Reranking {len(retrieved_docs)} documents...")
                    retrieved_docs = self._rerank_documents(question, retrieved_docs, top_k=reranker_top_k)

                source_docs = retrieved_docs

                # Format context from reranked documents
                context = self._extract_relevant_context(retrieved_docs, question)

                # Build the full prompt
                full_prompt = f"""You are a precise document analysis assistant. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
1. Answer based STRICTLY on the context provided - do not use external knowledge.
2. If the answer is clearly stated in the context, provide it accurately.
3. If you cannot find the answer in the context, say "I cannot find this information in the document."
4. Be specific and cite page numbers when available.

Previous conversation:
{history_text}

Context from document:
{context}

Question: {question}

Answer:"""

                # Get response from LLM
                response = self.llm.invoke(full_prompt, config={"callbacks": callbacks})
                result_text = response.content if hasattr(response, "content") else str(response)
            else:
                result_text = "This PDF hasn't been processed yet. Please upload and process it first."

        end_time = time.time()

        return {
            "result": result_text,
            "response_time": end_time - start_time,
            "source_documents": source_docs
        }

    def delete_pdf_index(self, pdf_file_path):
        """Delete vector store and cache for a PDF."""
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            del self._vector_store_cache[pdf_filename]

        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted vector store: {vector_store_path}")

        # Also delete vision index if exists
        vision_meta_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}_colpali_meta.pkl")
        vision_index_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_filename}_colpali.faiss")

        if os.path.exists(vision_meta_path):
            os.remove(vision_meta_path)
            logger.info(f"Deleted vision metadata: {vision_meta_path}")

        if os.path.exists(vision_index_path):
            os.remove(vision_index_path)
            logger.info(f"Deleted vision index: {vision_index_path}")

    def clear_cache(self):
        """Clear the vector store cache."""
        self._vector_store_cache.clear()
        logger.info("Vector store cache cleared")
