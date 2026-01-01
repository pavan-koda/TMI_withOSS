#!/bin/bash

# =============================================================================
# PDF Q&A System with 20B LLM - Setup Script
# Automated installation for Ubuntu 24
# =============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "PDF Q&A SYSTEM WITH 20B LLM - SETUP"
echo "================================================================================"
echo ""
echo "This script will install all dependencies for the PDF Q&A system."
echo "System Requirements:"
echo "  - RAM: 16GB+"
echo "  - GPU: NVIDIA with 4GB VRAM (optional, but recommended)"
echo "  - Storage: 20GB+ free"
echo "  - OS: Ubuntu 24 or similar Linux"
echo ""
read -p "Press Enter to continue..."

# =============================================================================
# Step 1: Check Python version
# =============================================================================

echo ""
echo "Step 1: Checking Python version..."
echo "-------------------------------------------------------------------------------"

python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python 3.9 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# =============================================================================
# Step 2: Create virtual environment
# =============================================================================

echo ""
echo "Step 2: Creating virtual environment..."
echo "-------------------------------------------------------------------------------"

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

echo "✓ Virtual environment created"

# =============================================================================
# Step 3: Upgrade pip
# =============================================================================

echo ""
echo "Step 3: Upgrading pip..."
echo "-------------------------------------------------------------------------------"

pip install --upgrade pip setuptools wheel

echo "✓ Pip upgraded"

# =============================================================================
# Step 4: Check for NVIDIA GPU and CUDA
# =============================================================================

echo ""
echo "Step 4: Checking for NVIDIA GPU..."
echo "-------------------------------------------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo "⚠ No NVIDIA GPU detected. Will install CPU-only version."
    echo "  (GPU offloading will NOT be available - inference will be slower)"
    HAS_GPU=false
fi

# =============================================================================
# Step 5: Install llama-cpp-python
# =============================================================================

echo ""
echo "Step 5: Installing llama-cpp-python..."
echo "-------------------------------------------------------------------------------"

if [ "$HAS_GPU" = true ]; then
    echo "Installing with CUDA support (this may take 5-10 minutes)..."
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.27 --force-reinstall --no-cache-dir --verbose
    echo "✓ llama-cpp-python installed with CUDA support"
else
    echo "Installing CPU-only version..."
    pip install llama-cpp-python==0.2.27
    echo "✓ llama-cpp-python installed (CPU-only)"
fi

# =============================================================================
# Step 6: Install other dependencies
# =============================================================================

echo ""
echo "Step 6: Installing other dependencies..."
echo "-------------------------------------------------------------------------------"

pip install -r requirements.txt

echo "✓ All dependencies installed"

# =============================================================================
# Step 7: Create necessary directories
# =============================================================================

echo ""
echo "Step 7: Creating directories..."
echo "-------------------------------------------------------------------------------"

mkdir -p uploads data logs processed_pdfs chroma_db models

echo "✓ Directories created"

# =============================================================================
# Step 8: Download model (optional)
# =============================================================================

echo ""
echo "Step 8: Model Download"
echo "-------------------------------------------------------------------------------"
echo "You need a GGUF model file to run this system."
echo ""
echo "Recommended models:"
echo "  1. Qwen2.5-14B-Instruct-Q4_K_M.gguf (~8GB)"
echo "  2. Mistral-7B-Instruct-v0.2-Q4_K_M.gguf (~4GB)"
echo "  3. Mixtral-8x7B-Instruct-Q4_K_M.gguf (~26GB)"
echo ""
echo "Download from: https://huggingface.co/TheBloke"
echo ""
read -p "Do you want to download a model now? (y/n): " download_model

if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo ""
    echo "Select model to download:"
    echo "  1. Qwen2.5-14B-Instruct-Q4_K_M (~8GB) - Recommended"
    echo "  2. Mistral-7B-Instruct-v0.2-Q4_K_M (~4GB) - Smaller, faster"
    echo "  3. Mixtral-8x7B-Instruct-Q4_K_M (~26GB) - Larger, better quality"
    read -p "Enter choice (1-3): " model_choice

    case $model_choice in
        1)
            MODEL_URL="https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct.Q4_K_M.gguf"
            MODEL_NAME="qwen2.5-14b-instruct.Q4_K_M.gguf"
            ;;
        2)
            MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            ;;
        3)
            MODEL_URL="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
            MODEL_NAME="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
            ;;
        *)
            echo "Invalid choice. Skipping download."
            MODEL_NAME=""
            ;;
    esac

    if [ -n "$MODEL_NAME" ]; then
        echo "Downloading $MODEL_NAME..."
        wget -O "models/$MODEL_NAME" "$MODEL_URL" || echo "Download failed. Please download manually."

        if [ -f "models/$MODEL_NAME" ]; then
            echo "✓ Model downloaded to models/$MODEL_NAME"
            echo ""
            echo "Updating model_config.py..."
            sed -i "s|models/gpt-20b-q4_k_m.gguf|models/$MODEL_NAME|g" model_config.py
            echo "✓ Configuration updated"
        fi
    fi
else
    echo "⚠ You will need to download a model manually and update model_config.py"
fi

# =============================================================================
# Step 9: Test installation
# =============================================================================

echo ""
echo "Step 9: Testing installation..."
echo "-------------------------------------------------------------------------------"

python3 model_config.py

# =============================================================================
# Installation Complete
# =============================================================================

echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. If you didn't download a model, download one from:"
echo "     https://huggingface.co/TheBloke"
echo "     and update model_config.py with the path"
echo ""
echo "  2. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  3. Start the application:"
echo "     python app_pdf_qa.py"
echo ""
echo "  4. Open browser:"
echo "     http://localhost:5000"
echo ""
echo "For GPU offloading (4GB VRAM), the default config uses 12 layers."
echo "Edit model_config.py to adjust n_gpu_layers if needed."
echo ""
echo "================================================================================"
