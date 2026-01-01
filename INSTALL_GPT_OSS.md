# Installing OpenAI gpt-oss-20b for PDF Q&A System

This guide covers installing and configuring **OpenAI's gpt-oss-20b** model (21B parameters, 3.6B active) for your PDF Q&A system.

---

## 🎯 Why gpt-oss-20b?

✅ **Perfect for your hardware**: 39GB RAM + 4GB VRAM
✅ **Efficient MoE architecture**: 21B params, only 3.6B active per token
✅ **MXFP4 quantization**: Fits in 16GB memory
✅ **Low latency**: Optimized for fast responses (5-15s target)
✅ **Apache 2.0 license**: Fully permissive, no restrictions
✅ **Built-in capabilities**: Function calling, reasoning levels, chain-of-thought

---

## 📦 Installation Options

### Option 1: Using Ollama (Easiest) ⭐ RECOMMENDED

**Fastest and simplest method**

```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull gpt-oss-20b model
ollama pull gpt-oss:20b

# 3. Verify installation
ollama list | grep gpt-oss

# 4. Test the model
ollama run gpt-oss:20b "What is quantum mechanics?"

# 5. Export model for use with llama-cpp-python
# Find model location (usually ~/.ollama/models/)
ls ~/.ollama/models/blobs/

# 6. Update model_config.py
nano model_config.py
# Set: "model_path": "~/.ollama/models/blobs/sha256-xxxxx"
# (or create a symlink to models/ directory)
```

**Advantages**:
- ✅ Automatic download and setup
- ✅ Pre-optimized for your system
- ✅ Easy to test before integration
- ✅ No manual GGUF conversion needed

---

### Option 2: Direct HuggingFace Download

**For advanced users who want more control**

```bash
# 1. Install HuggingFace CLI
pip install huggingface-hub

# 2. Download the model
huggingface-cli download openai/gpt-oss-20b \
  --include "original/*" \
  --local-dir models/gpt-oss-20b/

# 3. The model will be in PyTorch format, needs conversion to GGUF

# 4. Convert to GGUF format (if needed)
# Note: gpt-oss-20b uses MXFP4 quantization already
# You may need to use llama.cpp conversion tools
```

**Note**: This method downloads the raw PyTorch model. You may need to convert it to GGUF format for use with llama-cpp-python.

---

### Option 3: Using Transformers (Alternative)

**If you want to use HuggingFace Transformers directly**

```bash
# 1. Install dependencies
pip install transformers torch kernels

# 2. Model will auto-download on first use
python -c "from transformers import pipeline; \
  pipe = pipeline('text-generation', model='openai/gpt-oss-20b', torch_dtype='auto', device_map='auto'); \
  print('Model loaded successfully')"
```

**Note**: This uses Transformers instead of llama-cpp-python. You'll need to modify `pdf_qa_engine.py` to use Transformers API.

---

## ⚙️ Configuration for Your System

### Recommended Settings (39GB RAM + 4GB VRAM)

Edit [model_config.py](model_config.py):

```python
MODEL_CONFIG = {
    # Model path (after Ollama installation)
    "model_path": "~/.ollama/models/blobs/sha256-xxxxx",  # Find with: ls ~/.ollama/models/blobs/

    # Or if you created a symlink
    "model_path": "models/gpt-oss-20b.gguf",

    # GPU offloading for 4GB VRAM
    "n_gpu_layers": 15,  # Offload 15 layers to GPU (~2.5GB VRAM)

    # CPU threads
    "n_threads": 8,  # Use 8 CPU cores

    # Context window
    "n_ctx": 4096,  # 4K context window

    # Generation settings
    "max_tokens": 2048,
    "temperature": 0.3,  # Lower for more focused answers
    "top_p": 0.9,
    "top_k": 40,
}
```

### Memory Usage Breakdown

| Component | RAM | VRAM |
|-----------|-----|------|
| gpt-oss-20b model (MXFP4) | 16 GB | - |
| 15 layers on GPU | - | 2.5 GB |
| ChromaDB index | 1-2 GB | - |
| Flask + overhead | 2 GB | - |
| **Total** | **~20 GB** | **~2.5 GB** |

✅ Fits comfortably in your 39GB RAM + 4GB VRAM

---

## 🚀 Quick Start After Installation

### 1. Verify Model Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Test model configuration
python model_config.py
```

**Expected Output**:
```
================================================================================
PDF Q&A SYSTEM - MODEL CONFIGURATION
================================================================================
Model Path: models/gpt-oss-20b.gguf
Model Type: gpt-oss
Context Window: 4096 tokens
GPU Layers: 15
Estimated VRAM: 2.5 GB
CPU Threads: 8
Max Answer Length: 2048 tokens
================================================================================
Target Response Time: 15s
Confidence Scoring: True
Multi-Page Support: True
================================================================================

✅ Model file found!
```

### 2. Install System Dependencies

```bash
# Run setup script
chmod +x setup.sh
./setup.sh
```

### 3. Start Application

```bash
# Activate environment
source venv/bin/activate

# Start app
python app_pdf_qa.py
```

**Expected Output**:
```
================================================================================
PDF QA Engine initialized successfully
Model Info: {'model_path': 'models/gpt-oss-20b.gguf', 'n_gpu_layers': 15, ...}
================================================================================
PDF Q&A SYSTEM WITH 20B LLM
================================================================================
Server running at: http://localhost:5000
Analytics: http://localhost:5000/view-log
Health check: http://localhost:5000/health
================================================================================
```

### 4. Test the System

```bash
# In a new terminal, test health endpoint
curl http://localhost:5000/health

# Open browser
firefox http://localhost:5000
```

---

## 🎛️ Reasoning Levels

gpt-oss-20b supports adjustable reasoning effort. Configure in your prompts:

```python
# In pdf_qa_engine.py, modify the prompt:

# Low reasoning (fastest)
prompt = "Reasoning: low\n\nAnswer this question: {question}"

# Medium reasoning (balanced) - DEFAULT
prompt = "Reasoning: medium\n\nAnswer this question: {question}"

# High reasoning (detailed)
prompt = "Reasoning: high\n\nAnswer this question: {question}"
```

**Performance Impact**:
- **Low**: ~5-10 seconds per answer
- **Medium**: ~10-15 seconds per answer (default)
- **High**: ~15-30 seconds per answer

---

## 🔧 Troubleshooting

### Issue: Model file not found

```bash
# If using Ollama, find the model file
ls -lh ~/.ollama/models/blobs/

# Create symlink to models directory
ln -s ~/.ollama/models/blobs/sha256-xxxxx models/gpt-oss-20b.gguf

# Update model_config.py
nano model_config.py
# Set correct path
```

### Issue: CUDA out of memory

```python
# Reduce GPU layers in model_config.py
"n_gpu_layers": 10,  # Try 10 instead of 15
```

### Issue: Slow inference

```bash
# Verify GPU is being used
nvidia-smi

# Check GPU layers setting
grep "n_gpu_layers" model_config.py

# Should be 15 for 4GB VRAM
```

### Issue: Import error for llama_cpp

```bash
# Reinstall llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 📊 Expected Performance

### With 15 GPU Layers (4GB VRAM)

| Task | Time | Throughput |
|------|------|------------|
| PDF Processing (100 pages) | 30-60s | ~2 pages/sec |
| Index Creation | 5-10s | - |
| Question Answering | **5-15s** | 20-30 tokens/sec |

### CPU Only (0 GPU Layers)

| Task | Time | Throughput |
|------|------|------------|
| Question Answering | 30-60s | 8-12 tokens/sec |

**Recommendation**: Use 15 GPU layers for optimal performance

---

## 🎯 Model Features

### 1. Chain-of-Thought Reasoning
gpt-oss-20b provides full access to reasoning process:

```python
# Enable in prompt
"Show your reasoning step by step, then provide the final answer."
```

### 2. Function Calling
Built-in support for structured outputs:

```python
# Define tools/functions
tools = [
    {
        "name": "search_pdf",
        "description": "Search for information in PDF",
        "parameters": {...}
    }
]
```

### 3. Harmony Response Format
gpt-oss models use "harmony" format - automatically applied by our prompt builder.

---

## 📚 Additional Resources

- **Model Card**: https://huggingface.co/openai/gpt-oss-20b
- **Ollama Documentation**: https://ollama.com/library/gpt-oss
- **gpt-oss GitHub**: https://github.com/openai/gpt-oss (reference implementations)
- **Apache 2.0 License**: https://www.apache.org/licenses/LICENSE-2.0

---

## ✅ Verification Checklist

Before using the system, verify:

- [ ] Model downloaded (via Ollama or HuggingFace)
- [ ] Model path configured in `model_config.py`
- [ ] GPU layers set to 15 (for 4GB VRAM)
- [ ] llama-cpp-python installed with CUDA support
- [ ] `python model_config.py` runs without errors
- [ ] Application starts: `python app_pdf_qa.py`
- [ ] Health endpoint works: `curl http://localhost:5000/health`
- [ ] Can upload PDF and ask questions

---

## 🎉 You're Ready!

Your PDF Q&A system is now configured with **OpenAI's gpt-oss-20b**:

- ✅ 21B parameter MoE model
- ✅ Optimized for your 39GB RAM + 4GB VRAM
- ✅ 5-15 second response times
- ✅ Multi-page citations
- ✅ Confidence scores
- ✅ Apache 2.0 license

**Start using**: http://localhost:5000
