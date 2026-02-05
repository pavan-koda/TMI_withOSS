"""
Configuration file for the PDF Q&A System
Optimized for FAST response times
"""

# QA Engine Configuration
QA_CONFIG = {
    'mode': 'extractive',
    'use_advanced_qa': False,
    'use_full_context': False,  # SPEED: Use top chunks only
    'top_k_chunks': 3,  # SPEED: Fewer chunks
    'max_answer_length': 1500,  # SPEED: Shorter answers
}

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Fast, small model
}

# Reranker Configuration - ENABLED for quality
RERANKER_CONFIG = {
    'enabled': True,  # QUALITY: Enable reranking for better relevance
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'top_k': 5,
}

# Retrieval Configuration - Optimized for quality
RETRIEVAL_CONFIG = {
    'initial_k': 8,  # QUALITY: Retrieve more chunks for better context
    'summary_k': 10,  # QUALITY: Retrieve more chunks for summaries
    'summary_use_mmr': False,
    'use_mmr': False,
    'mmr_lambda': 0.5,
    'score_threshold': 0.0,
}

# Generator Model Configuration
GENERATOR_CONFIG = {
    'model_name': 'none',
    'use_generator': False,
}

# PDF Processing Configuration - Smaller chunks for faster processing
PDF_CONFIG = {
    'chunk_size': 600,  # SPEED: Smaller chunks
    'chunk_overlap': 100,  # SPEED: Less overlap
    'separators': ["\n\n", "\n", ". ", " "],
}

# Flask Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,  # SPEED: Disable debug
    'max_content_length': 16 * 1024 * 1024,
}
