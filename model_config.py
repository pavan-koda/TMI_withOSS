"""
Model Configuration for PDF Q&A System with 20B LLM
Supports CPU + GPU offloading for efficient inference
"""

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Model file path (GGUF format)
    # Download from: https://huggingface.co/TheBloke
    # Recommended: GPT-20B-Q4_K_M.gguf or similar quantized model
    "model_path": os.getenv("MODEL_PATH", "models/gpt-20b-q4_k_m.gguf"),

    # Model type
    "model_type": "llama",  # Architecture: llama, gpt-neox, etc.

    # Context window
    "n_ctx": 4096,  # Maximum context length (tokens)

    # GPU offloading (CRITICAL for 4GB VRAM)
    # Set this based on your GPU memory:
    # - 0: CPU only (slow but works)
    # - 10-15: Good balance for 4GB VRAM (recommended)
    # - 20-25: If you have 6GB VRAM
    # - 35+: If you have 8GB+ VRAM
    "n_gpu_layers": int(os.getenv("N_GPU_LAYERS", "12")),  # Offload 12 layers to GPU

    # CPU threads
    "n_threads": int(os.getenv("N_THREADS", "8")),  # Use 8 CPU cores

    # Batch size
    "n_batch": 512,  # Tokens processed in parallel

    # Generation settings
    "max_tokens": 2048,  # Maximum answer length
    "temperature": 0.3,  # Lower = more focused, higher = more creative (0.0-1.0)
    "top_p": 0.9,  # Nucleus sampling
    "top_k": 40,  # Top-k sampling
    "repeat_penalty": 1.1,  # Penalize repetition

    # Memory optimization
    "use_mmap": True,  # Memory-map model file (saves RAM)
    "use_mlock": False,  # Lock model in RAM (set True if you have enough RAM)
    "low_vram": True,  # Enable for GPUs with <6GB VRAM
}

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

INFERENCE_CONFIG = {
    # Response time target (seconds)
    "target_response_time": 15,  # 5-15 seconds acceptable

    # Confidence score calculation
    "use_confidence_scoring": True,
    "confidence_method": "perplexity",  # Method: perplexity, logprobs, similarity

    # Page number tracking
    "track_page_numbers": True,
    "multi_page_support": True,  # Support answers from multiple pages
    "max_pages_per_answer": 5,  # Maximum pages to cite in one answer
}

# ============================================================================
# PDF PROCESSING SETTINGS
# ============================================================================

PDF_CONFIG = {
    # DPI for page rendering
    "dpi": 150,  # Options: 100, 150, 200, 300

    # PDF size limits
    "max_file_size_mb": 50,  # Maximum PDF size
    "max_pages": 1000,  # Maximum number of pages

    # Text extraction
    "extract_text": True,
    "extract_images": True,

    # Chunking strategy
    "chunk_size": 512,  # Tokens per chunk
    "chunk_overlap": 50,  # Overlap between chunks
}

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

RETRIEVAL_CONFIG = {
    # Number of pages to retrieve
    "top_k": 5,

    # Retrieval method
    "use_semantic_search": True,  # Use embeddings for search
    "use_keyword_search": True,  # Also use keyword matching

    # Embedding model
    "embedding_model": "all-MiniLM-L6-v2",  # Fast and accurate

    # Vector store
    "vector_store": "chroma",  # Options: chroma, faiss
    "persist_directory": "chroma_db",
}

# ============================================================================
# ANALYTICS & LOGGING
# ============================================================================

ANALYTICS_CONFIG = {
    # Log all Q&A sessions
    "enable_logging": True,
    "log_file": "logs/qa_performance.txt",

    # Track metrics
    "track_response_time": True,
    "track_confidence": True,
    "track_page_references": True,

    # Session management
    "session_timeout": 3600,  # 1 hour
    "max_sessions": 100,  # Maximum concurrent sessions
}

# ============================================================================
# SYSTEM REQUIREMENTS
# ============================================================================

SYSTEM_REQUIREMENTS = {
    "ram_gb": 39,
    "gpu_vram_gb": 4,
    "storage_gb": 45,
    "os": "Ubuntu 24",

    # Estimated memory usage (4-bit quantized 20B model)
    "model_ram_usage_gb": 12,  # Model loaded in RAM
    "model_vram_usage_gb": 2,  # 12 layers on GPU (~2GB)
    "total_ram_needed_gb": 16,  # Model + ChromaDB + Flask
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path():
    """Get absolute model path."""
    model_path = Path(MODEL_CONFIG["model_path"])

    if not model_path.is_absolute():
        # If relative path, make it relative to this config file
        config_dir = Path(__file__).parent
        model_path = config_dir / model_path

    return str(model_path)

def validate_model_exists():
    """Check if model file exists."""
    model_path = get_model_path()

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n\n"
            f"Please download a GGUF model and update model_config.py.\n"
            f"Recommended models:\n"
            f"  - GPT-20B-Q4_K_M.gguf (good quality, ~12GB)\n"
            f"  - Mistral-7B-Instruct-v0.2.Q4_K_M.gguf (smaller, ~4GB)\n"
            f"  - Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf (better quality, ~26GB)\n\n"
            f"Download from: https://huggingface.co/TheBloke"
        )

    return True

def estimate_vram_usage():
    """Estimate VRAM usage based on n_gpu_layers."""
    layers = MODEL_CONFIG["n_gpu_layers"]

    # Rough estimate: ~150MB per layer for 20B model
    vram_gb = (layers * 150) / 1024

    return round(vram_gb, 2)

def get_recommended_layers_for_vram(vram_gb):
    """Get recommended n_gpu_layers for given VRAM."""
    if vram_gb <= 2:
        return 0  # CPU only
    elif vram_gb <= 4:
        return 12  # ~2GB VRAM usage
    elif vram_gb <= 6:
        return 20  # ~3GB VRAM usage
    elif vram_gb <= 8:
        return 35  # ~5GB VRAM usage
    else:
        return 60  # Full offload for 12GB+ VRAM

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration."""
    print("="*80)
    print("PDF Q&A SYSTEM - MODEL CONFIGURATION")
    print("="*80)
    print(f"Model Path: {get_model_path()}")
    print(f"Model Type: {MODEL_CONFIG['model_type']}")
    print(f"Context Window: {MODEL_CONFIG['n_ctx']} tokens")
    print(f"GPU Layers: {MODEL_CONFIG['n_gpu_layers']}")
    print(f"Estimated VRAM: {estimate_vram_usage()} GB")
    print(f"CPU Threads: {MODEL_CONFIG['n_threads']}")
    print(f"Max Answer Length: {MODEL_CONFIG['max_tokens']} tokens")
    print("="*80)
    print(f"Target Response Time: {INFERENCE_CONFIG['target_response_time']}s")
    print(f"Confidence Scoring: {INFERENCE_CONFIG['use_confidence_scoring']}")
    print(f"Multi-Page Support: {INFERENCE_CONFIG['multi_page_support']}")
    print("="*80)

if __name__ == "__main__":
    print_config()

    try:
        validate_model_exists()
        print("\n✅ Model file found!")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
