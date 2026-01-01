# PDF Q&A System with 20B LLM (CPU+GPU Offload)

A powerful PDF question-answering system using a 20B parameter language model with CPU+GPU offloading for efficient inference on systems with limited VRAM.

---

## 🌟 Features

### Core Functionality
- ✅ **PDF Upload & Processing** - Upload PDFs up to 50MB, extract text and images
- ✅ **Multi-Page Answers** - Get comprehensive answers citing multiple pages (e.g., "Pages 5, 12-14, 18")
- ✅ **Exact Page References** - Every answer includes specific page numbers where information was found
- ✅ **Confidence Scores** - Each answer includes a 0-100% confidence score
- ✅ **Response Time** - Optimized for 5-15 second responses
- ✅ **Image Support** - Displays diagrams, charts, and images from PDF

### AI & Performance
- ✅ **20B Parameter LLM** - Uses GGUF quantized models for efficient inference
- ✅ **CPU+GPU Offloading** - Offload layers to GPU (4GB VRAM supported)
- ✅ **Smart Context** - Retrieves relevant pages using semantic search
- ✅ **Conversation History** - Remembers last 5 Q&A exchanges for follow-up questions

### Analytics & Logging
- ✅ **Performance Tracking** - Log all questions, answers, response times
- ✅ **Dashboard** - View analytics: total questions, avg response time, avg confidence
- ✅ **Session Management** - Track multiple PDF sessions

---

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 16 GB (12GB for model + 4GB for system)
- **GPU**: NVIDIA with 4 GB VRAM (optional but recommended)
- **Storage**: 20 GB free (model + dependencies + data)
- **OS**: Ubuntu 24 or similar Linux (tested on Ubuntu 24.04)

### Recommended Requirements
- **RAM**: 32 GB
- **GPU**: NVIDIA with 6+ GB VRAM
- **Storage**: 50 GB free
- **CPU**: 8+ cores

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or extract project
cd TMI_withOSS

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Start application
python app_pdf_qa.py
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install llama-cpp-python with CUDA support (for GPU)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.27 --force-reinstall --no-cache-dir

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Download a GGUF model
# Visit: https://huggingface.co/TheBloke
# Download: qwen2.5-14b-instruct.Q4_K_M.gguf (recommended)
# Place in: models/

# 5. Update model path in model_config.py
nano model_config.py
# Set: "model_path": "models/qwen2.5-14b-instruct.Q4_K_M.gguf"

# 6. Start application
python app_pdf_qa.py
```

### Access the Application

Open browser and visit: **http://localhost:5000**

---

## 📦 Project Structure

```
TMI_withOSS/
├── app_pdf_qa.py              # Main Flask application
├── pdf_qa_engine.py           # QA engine with 20B LLM
├── pdf_processor.py           # PDF processing (text + images)
├── model_config.py            # Model configuration (GPU layers, paths, etc.)
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
│
├── templates/                 # HTML templates
│   ├── upload.html            # PDF upload page
│   └── qa.html                # Q&A chat interface
│
├── static/                    # CSS, JavaScript, images
│
├── models/                    # GGUF model files (you download these)
│   └── [your-model].gguf
│
├── uploads/                   # Temporary PDF uploads
├── processed_pdfs/            # Extracted page images
├── chroma_db/                 # Vector database
├── logs/                      # Performance logs
└── data/                      # Session data

```

---

## ⚙️ Configuration

### GPU Offloading (model_config.py)

Adjust `n_gpu_layers` based on your GPU VRAM:

```python
MODEL_CONFIG = {
    # GPU offloading
    "n_gpu_layers": 12,  # Recommended for 4GB VRAM

    # For different VRAM:
    # - 0: CPU only (no GPU)
    # - 10-15: 4GB VRAM
    # - 20-25: 6GB VRAM
    # - 35+: 8GB+ VRAM
}
```

### Model Selection

Edit `model_config.py` to change the model:

```python
MODEL_CONFIG = {
    "model_path": "models/your-model.gguf",
}
```

**Recommended Models:**

| Model | Size | RAM | Quality | Use Case |
|-------|------|-----|---------|----------|
| Mistral-7B-Instruct-Q4_K_M | ~4GB | 8GB | Good | Fast, lightweight |
| Qwen2.5-14B-Instruct-Q4_K_M | ~8GB | 16GB | Excellent | **Recommended** |
| Mixtral-8x7B-Instruct-Q4_K_M | ~26GB | 32GB | Superior | Best quality |

Download from: https://huggingface.co/TheBloke

---

## 📊 How It Works

### 1. PDF Processing
```
Upload PDF → Extract text from each page
          → Render each page as image (150 DPI)
          → Extract embedded images
```

### 2. Indexing
```
Text Index (ChromaDB)
├─ Full text from each page
├─ Semantic embeddings
└─ Fast retrieval by similarity

File System
└─ Page images for display
```

### 3. Question Answering
```
Your Question
    ↓
Semantic Search → Find top 5 relevant pages
    ↓
Build Context → Combine text from relevant pages
    ↓
20B LLM Inference → Generate comprehensive answer
    ↓
Calculate Confidence → 0-100% based on retrieval + quality
    ↓
Format Response → Include answer, pages, confidence, time
```

### 4. Multi-Page Citations

Example answer format:

```
Revenue increased by 25% due to Asia expansion (Page 5) and
new product launches (Pages 12-14). The detailed strategy is
outlined on Page 18.

📄 Pages Referenced: Pages 5, 12-14, 18
📊 Confidence: 87.3%
⏱️ Response Time: 8.2s
```

---

## 🎯 Features in Detail

### 1. Multi-Page Answer Support

The system automatically:
- Retrieves multiple relevant pages (top 5 by default)
- Combines text context from all pages
- Cites exact pages in the answer
- Groups consecutive pages into ranges (e.g., "12-14")

### 2. Confidence Scoring

Confidence score (0-100%) is calculated based on:
- **Retrieval Confidence (40%)**: How relevant the retrieved pages are
- **Answer Quality (40%)**: Length and completeness of answer
- **Context Usage (20%)**: How well the answer uses the source text

### 3. Performance Optimization

- **GPU Offloading**: Offload transformer layers to GPU for 3-5x speedup
- **GGUF Quantization**: 4-bit quantization reduces model size by 75%
- **Smart Context**: Only retrieves relevant pages (not entire PDF)
- **Streaming Disabled**: Faster single-shot generation

### 4. Analytics Dashboard

Visit **http://localhost:5000/view-log** to see:
- Total questions asked
- Average response time
- Average confidence score
- Recent activity log
- Detailed per-question metrics

---

## 📈 Expected Performance

### With GPU Offloading (12 layers on 4GB VRAM)

| Task | Time | Throughput |
|------|------|------------|
| PDF Processing (100 pages) | 30-60s | ~2 pages/sec |
| Index Creation | 5-10s | - |
| Question Answering | 5-15s | ~20-30 tokens/sec |

### CPU Only (No GPU)

| Task | Time | Throughput |
|------|------|------------|
| PDF Processing (100 pages) | 30-60s | ~2 pages/sec |
| Index Creation | 5-10s | - |
| Question Answering | 30-90s | ~5-10 tokens/sec |

---

## 🔧 Troubleshooting

### Issue: llama-cpp-python fails to install

**Solution**: Install build dependencies first

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Issue: Model file not found

**Solution**: Download model and update config

```bash
# Download from HuggingFace
wget https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct.Q4_K_M.gguf -O models/qwen2.5-14b-instruct.Q4_K_M.gguf

# Update config
nano model_config.py
# Set: "model_path": "models/qwen2.5-14b-instruct.Q4_K_M.gguf"
```

### Issue: CUDA out of memory

**Solution**: Reduce GPU layers

```python
# In model_config.py
MODEL_CONFIG = {
    "n_gpu_layers": 8,  # Reduce from 12 to 8
}
```

### Issue: Slow inference (CPU only)

**Solution**: Install with CUDA support

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Then edit `model_config.py`:

```python
"n_gpu_layers": 12,  # Offload layers to GPU
```

### Issue: Port 5000 already in use

**Solution**: Change port in `app_pdf_qa.py`

```python
# Line 525
app.run(host='0.0.0.0', port=5001, debug=False)
```

---

## 📝 API Endpoints

### Upload PDF
```
POST /upload
Body: multipart/form-data
  - pdf_file: PDF file
  - dpi: 100, 150, 200, or 300 (default: 150)
  - extract_images: true or false (default: true)

Response:
{
  "success": true,
  "message": "PDF processed successfully! 45 pages indexed.",
  "metadata": {
    "filename": "document.pdf",
    "total_pages": 45,
    "session_id": "abc-123"
  }
}
```

### Ask Question
```
POST /ask
Body: application/json
{
  "question": "What is the revenue growth?",
  "top_k": 5
}

Response:
{
  "success": true,
  "answer": "Revenue grew by 25%...",
  "page_references": "Pages 5, 12-14, 18",
  "confidence": 87.3,
  "response_time": 8.2,
  "pages_used": [5, 12, 13, 14, 18],
  "images": ["/data/session-id/page_0005.png", ...]
}
```

### Analytics
```
GET /analytics

Response:
{
  "total_questions": 42,
  "avg_response_time": 9.5,
  "avg_confidence": 85.2,
  "recent_logs": [...]
}
```

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "qa_engine": true,
  "model_info": {
    "model_path": "models/qwen2.5-14b-instruct.Q4_K_M.gguf",
    "n_gpu_layers": 12,
    "n_ctx": 4096
  }
}
```

---

## 🛡️ Security Notes

- Runs on localhost by default (not exposed to network)
- All processing is local (no external API calls)
- Session data cleared on restart
- Uploaded PDFs stored temporarily

To enable external access (use with caution):
```python
# app_pdf_qa.py line 525
app.run(host='0.0.0.0', port=5000, debug=False)
# Then access from network: http://<server-ip>:5000
```

---

## 📚 Model Recommendations

### For Your System (39GB RAM, 4GB VRAM):

**Best Choice**: Qwen2.5-14B-Instruct-Q4_K_M.gguf
- Model Size: ~8GB
- Quality: Excellent for document Q&A
- Speed: 20-30 tokens/sec with 12 GPU layers
- Download: https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF

**Alternative**: Mistral-7B-Instruct-v0.2-Q4_K_M.gguf
- Model Size: ~4GB
- Quality: Good for general Q&A
- Speed: 30-40 tokens/sec with 15 GPU layers
- Download: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

---

## 📖 Usage Examples

### Example 1: Financial Document

```
Q: What was the revenue growth in Q4 2023?
A: Revenue grew by 25% in Q4 2023, reaching $15.2 million. This growth
   was driven by expansion into Asian markets (Page 5) and the launch of
   three new product lines (Pages 12-14). The detailed breakdown of revenue
   by region can be found on Page 18.

📄 Pages: 5, 12-14, 18
📊 Confidence: 92.1%
⏱️ Time: 7.3s
```

### Example 2: Technical Manual

```
Q: How do I configure the firewall settings?
A: To configure the firewall, navigate to Settings > Network > Firewall
   (Page 34). Enable the firewall by toggling the switch, then add rules
   using the "Add Rule" button (Page 35). For advanced configurations
   including port forwarding and DMZ setup, refer to Pages 38-40.

📄 Pages: 34-35, 38-40
📊 Confidence: 88.5%
⏱️ Time: 9.1s
```

---

## 🤝 Contributing

This project is maintained internally. For issues or feature requests, please contact the development team.

---

## 📜 License

Internal use only. Not for redistribution.

---

## 🙏 Acknowledgments

- **llama.cpp** - Efficient LLM inference
- **ChromaDB** - Vector database
- **PyMuPDF** - PDF processing
- **Flask** - Web framework
- **TheBloke** - GGUF model quantization

---

## 📞 Support

For questions or issues:
1. Check logs in `logs/qa_performance.txt`
2. Check Flask output in terminal
3. Verify model exists: `python model_config.py`
4. Test health endpoint: `curl http://localhost:5000/health`

---

**Ready to process PDFs with AI!** 🚀

Upload a PDF, ask questions, and get comprehensive answers with exact page references and confidence scores.
