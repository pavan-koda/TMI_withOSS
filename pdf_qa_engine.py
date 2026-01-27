from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.schema import Document
import logging
import time
import os
import re
import shutil

# Try to import CrossEncoder for reranking
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logging.warning("CrossEncoder not available. Install sentence-transformers for reranking support.")

# Import config with fallback defaults
try:
    from config import EMBEDDING_CONFIG, PDF_CONFIG
except ImportError:
    EMBEDDING_CONFIG = {'model_name': 'BAAI/bge-base-en-v1.5'}
    PDF_CONFIG = {'chunk_size': 800, 'chunk_overlap': 200}

try:
    from config import RERANKER_CONFIG
except (ImportError, KeyError):
    RERANKER_CONFIG = {
        'enabled': True,
        'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'top_k': 5
    }

try:
    from config import RETRIEVAL_CONFIG
except (ImportError, KeyError):
    RETRIEVAL_CONFIG = {
        'initial_k': 10,
        'use_mmr': True,
        'mmr_lambda': 0.7,
        'score_threshold': 0.3
    }

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_stores"


class PDFQAEngine:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, streaming=True)

        # Use embedding model from config
        embedding_model = EMBEDDING_CONFIG.get('model_name', 'all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model}")

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model}: {e}")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize cross-encoder reranker for improved accuracy
        self.reranker = None
        if CROSSENCODER_AVAILABLE and RERANKER_CONFIG.get('enabled', True):
            try:
                reranker_model = RERANKER_CONFIG.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info(f"Loading reranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")

        self._vector_store_cache = {}
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)

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
        vector_store_path = self._get_vector_store_path(pdf_file_path)

        if not os.path.exists(vector_store_path):
            logger.info(f"Ingesting PDF: {pdf_file_path}")
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
                logger.warning(f"No text found in {pdf_file_path}. Creating placeholder.")
                chunks = [Document(
                    page_content="[This document contains no extractable text.]",
                    metadata={"source": pdf_file_path, "page": 0}
                )]

            # Skip enhancing chunks with context to avoid slowing down embedding and text processing
            # chunks = self._enhance_chunks_with_context(chunks, pdf_file_path)

            logger.info(f"Created {len(chunks)} chunks from PDF")

            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to {vector_store_path}")

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

            logger.info(f"Reranking - Top scores: {[f'{s:.3f}' for _, s in scored_docs[:3]]}")

            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return documents[:top_k]

    def _preprocess_query(self, query):
        """Preprocess query for better matching."""
        # Just clean up whitespace, don't remove words
        return query.strip()

    def _is_greeting(self, text):
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        text = text.lower().strip(" .!,")
        return text in greetings

    def _extract_relevant_context(self, documents, query):
        """Extract and format the most relevant parts of retrieved documents."""
        context_parts = []
        seen_content = set()

        for doc in documents:
            content = doc.metadata.get('original_content', doc.page_content)

            # Avoid duplicate content
            content_hash = hash(content[:100] if len(content) > 100 else content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            # Only append content, do not include page numbers in the context text passed to LLM
            context_parts.append(f"{content}")

        return "\n\n---\n\n".join(context_parts)

    def _load_vector_store(self, pdf_file_path):
        """Load vector store for a PDF, with caching."""
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
                return vector_store
            return None

        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            return self._vector_store_cache[pdf_filename]

        if os.path.exists(vector_store_path):
            try:
                vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._vector_store_cache[pdf_filename] = vector_store
                return vector_store
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")

        return None

    def answer_question(self, question, pdf_file_path, callbacks=None, use_vision=False, chat_history=None):
        """Answer question with improved retrieval and reranking."""
        start_time = time.time()

        if self._is_greeting(question):
            return {
                "result": "Hello! I'm ready to help you with your document. You can ask me:\n\n- **Summary questions** - \"What is this document about?\"\n- **Specific details** - \"What are the key points?\"\n- **Search queries** - \"Find information about...\"\n\nWhat would you like to know?",
                "response_time": 0.0,
                "source_documents": []
            }

        if chat_history is None:
            chat_history = []

        # Format chat history for context (last 4 messages)
        history_text = ""
        for msg in chat_history[-4:]:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if role and content:
                history_text += f"{role}: {content}\n"

        # Preprocess the query
        processed_query = self._preprocess_query(question)

        # Load vector store
        vector_store = self._load_vector_store(pdf_file_path)

        if not vector_store:
            return {
                "result": "This PDF hasn't been processed yet. Please upload and process it first.",
                "response_time": time.time() - start_time,
                "source_documents": []
            }

        logger.info("Answering question from PDF...")

        # Get retrieval settings
        initial_k = RETRIEVAL_CONFIG.get('initial_k', 8)
        reranker_top_k = RERANKER_CONFIG.get('top_k', 5)

        # Retrieve documents - use simple similarity search for reliability
        retrieved_docs = []
        try:
            logger.info(f"Searching for: {processed_query}")
            retrieved_docs = vector_store.similarity_search(processed_query, k=initial_k)
            logger.info(f"Found {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")

        if not retrieved_docs:
            # Try with original question if processed query failed
            try:
                retrieved_docs = vector_store.similarity_search(question, k=initial_k)
                logger.info(f"Retry found {len(retrieved_docs)} documents")
            except Exception as e:
                logger.error(f"Retry retrieval failed: {e}")

        if not retrieved_docs:
            return {
                "result": "I couldn't find relevant information for your question. Please make sure the PDF has been indexed properly.",
                "response_time": time.time() - start_time,
                "source_documents": []
            }

        # Apply reranking for better relevance
        if self.reranker and retrieved_docs:
            logger.info(f"Reranking {len(retrieved_docs)} documents...")
            retrieved_docs = self._rerank_documents(question, retrieved_docs, top_k=reranker_top_k)

        source_docs = retrieved_docs

        # Format context from reranked documents
        context = self._extract_relevant_context(retrieved_docs, question)

        # Build the prompt
        prompt = f"""You are a helpful document assistant. Read the context carefully and answer the user's question.

RULES:
1. ONLY use information from the context below - never make up information
2. Give clear, direct answers - get to the point quickly
3. Use bullet points for multiple items
4. Include specific details like numbers, dates, names when available
5. If the answer has multiple parts, organize them clearly
6. If information is NOT in the context, say: "This information is not available in the document."
7. DO NOT mention page numbers or source documents in the text of your answer.

FORMAT YOUR RESPONSE:
- Keep answers concise but complete
- Use short paragraphs (2-3 sentences max)
- For lists, use bullet points
- Bold **key terms** if helpful
 - Do NOT include citations or page references in the text.

Previous conversation:
{history_text}

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

        # Get response from LLM
        try:
            response = self.llm.invoke(prompt, config={"callbacks": callbacks})
            result_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            result_text = f"Error generating response: {str(e)}"

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

    def clear_cache(self):
        """Clear the vector store cache."""
        self._vector_store_cache.clear()
        logger.info("Vector store cache cleared")

    def rebuild_index(self, pdf_file_path, use_vision=False):
        """Force rebuild the vector store index for a PDF."""
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        pdf_filename = os.path.basename(pdf_file_path)

        if pdf_filename in self._vector_store_cache:
            del self._vector_store_cache[pdf_filename]

        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            logger.info(f"Removed old index: {vector_store_path}")

        self.ingest_pdf(pdf_file_path)
        logger.info(f"Rebuilt index for: {pdf_filename}")

    def is_pdf_indexed(self, pdf_file_path):
        """Check if a PDF has been indexed."""
        vector_store_path = self._get_vector_store_path(pdf_file_path)
        return os.path.exists(vector_store_path)
