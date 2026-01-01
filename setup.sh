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
echo "This system is configured for OpenAI's gpt-oss-20b model."
echo ""
echo "Model Info:"
echo "  - gpt-oss-20b: 21B params, 3.6B active (MoE architecture)"
echo "  - Memory: ~16GB RAM with MXFP4 quantization"
echo "  - License: Apache 2.0 (fully permissive)"
echo "  - Perfect for: 39GB RAM + 4GB VRAM systems"
echo ""
echo "Installation options:"
echo "  1. Ollama (Easiest) - Recommended"
echo "  2. Direct download from HuggingFace"
echo "  3. Skip (manual installation later)"
echo ""
read -p "Choose installation method (1-3): " download_model

case $download_model in
    1)
        # Ollama installation
        echo ""
        echo "Installing via Ollama..."
        echo "-------------------------------------------------------------------------------"

        # Check if Ollama is installed
        if ! command -v ollama &> /dev/null; then
            echo "Ollama not found. Installing Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh
        else
            echo "✓ Ollama already installed"
        fi

        # Pull gpt-oss-20b model
        echo ""
        echo "Downloading gpt-oss-20b model (this may take 5-10 minutes)..."
        ollama pull gpt-oss:20b

        if [ $? -eq 0 ]; then
            echo "✓ Model downloaded successfully"

            # Find model location
            OLLAMA_MODELS_DIR="$HOME/.ollama/models/blobs"
            if [ -d "$OLLAMA_MODELS_DIR" ]; then
                echo ""
                echo "Model stored in: $OLLAMA_MODELS_DIR"

                # Find the latest blob (model file)
                LATEST_BLOB=$(ls -t "$OLLAMA_MODELS_DIR"/sha256-* 2>/dev/null | head -1)

                if [ -n "$LATEST_BLOB" ]; then
                    # Create symlink in models directory
                    ln -sf "$LATEST_BLOB" models/gpt-oss-20b.gguf
                    echo "✓ Created symlink: models/gpt-oss-20b.gguf -> $LATEST_BLOB"

                    # Update config
                    sed -i 's|models/gpt-oss-20b.gguf|models/gpt-oss-20b.gguf|g' model_config.py
                    echo "✓ Configuration updated"
                else
                    echo "⚠ Could not find model file. Please update model_config.py manually."
                    echo "  Model location: $OLLAMA_MODELS_DIR"
                fi
            fi
        else
            echo "✗ Failed to download model via Ollama"
            echo "  Please try manual installation (see INSTALL_GPT_OSS.md)"
        fi
        ;;

    2)
        # Direct HuggingFace download
        echo ""
        echo "Downloading from HuggingFace..."
        echo "-------------------------------------------------------------------------------"

        # Install huggingface-cli if not present
        pip install -q huggingface-hub

        # Download model
        echo "Downloading gpt-oss-20b (this may take 10-20 minutes)..."
        huggingface-cli download openai/gpt-oss-20b \
            --include "original/*" \
            --local-dir models/gpt-oss-20b/

        if [ $? -eq 0 ]; then
            echo "✓ Model downloaded to models/gpt-oss-20b/"
            echo ""
            echo "⚠ Note: Model is in PyTorch format"
            echo "  You may need to convert to GGUF format for llama-cpp-python"
            echo "  Or use Transformers directly (modify pdf_qa_engine.py)"
            echo ""
            echo "  See INSTALL_GPT_OSS.md for conversion instructions"
        else
            echo "✗ Failed to download from HuggingFace"
        fi
        ;;

    3)
        # Skip download
        echo ""
        echo "⚠ Skipping model download"
        echo ""
        echo "To install gpt-oss-20b later, run:"
        echo "  ollama pull gpt-oss:20b"
        echo ""
        echo "Or see INSTALL_GPT_OSS.md for detailed instructions"
        ;;

    *)
        echo "Invalid choice. Skipping download."
        ;;
esac

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
