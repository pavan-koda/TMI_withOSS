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

        # Use better embedding model from config
        embedding_model = EMBEDDING_CONFIG.get('model_name', 'BAAI/bge-base-en-v1.5')
        logger.info(f"Loading embedding model: {embedding_model}")

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.warning(f"Failed to load {embedding_model}: {e}. Falling back to all-MiniLM-L6-v2")
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

            # Enhance chunks with contextual information
            chunks = self._enhance_chunks_with_context(chunks, pdf_file_path)

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
        query = query.strip()

        # Remove filler words that don't add semantic meaning
        query = re.sub(r'\b(please|kindly|can you|could you|tell me|what is|what are)\b', '', query, flags=re.IGNORECASE)
        query = query.strip()

        if len(query.split()) <= 2:
            return query

        return query

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

            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(f"[Page {page + 1 if isinstance(page, int) else page}]: {content}")

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
                "result": "Hello! How can I help you with your document today?",
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

        logger.info("Using enhanced Text RAG with reranking.")

        # Get retrieval settings
        initial_k = RETRIEVAL_CONFIG.get('initial_k', 10)
        use_mmr = RETRIEVAL_CONFIG.get('use_mmr', True)
        mmr_lambda = RETRIEVAL_CONFIG.get('mmr_lambda', 0.7)
        reranker_top_k = RERANKER_CONFIG.get('top_k', 5)

        # Retrieve documents using MMR for diversity
        try:
            if use_mmr:
                retrieved_docs = vector_store.max_marginal_relevance_search(
                    processed_query,
                    k=initial_k,
                    lambda_mult=mmr_lambda,
                    fetch_k=initial_k * 2
                )
            else:
                retrieved_docs = vector_store.similarity_search(processed_query, k=initial_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved_docs = []

        if not retrieved_docs:
            return {
                "result": "I couldn't find relevant information in the document for your question.",
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
        prompt = f"""You are a precise document analysis assistant. Answer the question based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. Read the context carefully and thoroughly before answering.
2. Base your answer STRICTLY on the information in the context - do not add external knowledge.
3. If the exact answer is in the context, quote or paraphrase it accurately.
4. If the context contains partial information, provide what is available and note what's missing.
5. If the answer is NOT in the context, respond: "I cannot find this information in the provided document."
6. For numerical data, dates, or specific facts - be precise and cite the page number.
7. If multiple pieces of context are relevant, synthesize them into a coherent answer.

Previous conversation:
{history_text}

Context from document:
{context}

Question: {question}

Answer:"""

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
