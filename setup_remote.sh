#!/usr/bin/env bash
# ---------------------------------------------------------------
# Setup script for the remote GPU machine
#
# Usage:
#   git clone <your-repo> && cd self_supervised_tracking
#   chmod +x setup_remote.sh
#   ./setup_remote.sh
#
# After setup:
#   source venv/bin/activate
#   python train.py                    # train with defaults
#   python train.py --epochs 100       # custom epochs
#   python evaluate.py                 # evaluate best model
# ---------------------------------------------------------------

set -e

echo "=== Setting up self-supervised particle tracking ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (adjust cu121 to match your CUDA version)
# Check yours with: nvcc --version  or  nvidia-smi
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
pip install numpy matplotlib tensorboard

# Verify
echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import numpy; print(f'NumPy {numpy.__version__}')
import matplotlib; print(f'Matplotlib {matplotlib.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate with:  source venv/bin/activate"
echo "Train with:     python train.py"
echo "Evaluate with:  python evaluate.py"
echo ""
echo "=== Live visualisation ==="
echo "On the GPU machine:  tensorboard --logdir runs/ --port 6006"
echo "On your laptop:      ssh -L 6006:localhost:6006 user@gpu-machine"
echo "Then open:           http://localhost:6006"
