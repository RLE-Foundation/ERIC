<img src="https://github.com/RLE-Foundation/ERIC/blob/main/docs/eric_logo.svg">


<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/Base-PyTorch-EF4B28"> <img src="https://img.shields.io/badge/Code%20style-Black-000000"> <img src="https://img.shields.io/badge/Python-%3E%3D3.10-%2335709F"> <a href="https://discord.gg/YGApGaXAHW"><img src="https://img.shields.io/badge/Discussion-Discord-5562EA" alt="Discussion Discord"></a> <a href="https://arxiv.org/pdf/2504.17490"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a> 

---

**Embodied Reinforcement IntelligenCe (ERIC)** is a framework that provides high-quality single-file implementations for finetuning vision-language-action (VLA) models via reinforcement learning. Following the design philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl), ERIC is clean and simple, accelerating your research with user-friendly features. The highlight features of CleanRL are:
- 📜 Single-file implementation

# Finetuning VLA with RL from Scratch

First time in this area? 

Don't worry, we provide a great notebook that helps you understand this area and build your project step by step!

See [Fintuning VLA with RL from Scratch]().

# Quick Start

### Prerequisites

- **Python**: 3.10 (recommended)
- **CUDA**: 11.8+ or 12.1+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ for training)

### Installation

```bash
# 1. Create conda environment
conda create -n eric python=3.10
conda activate eric

# 2. Install LIBERO from source
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/

# 3. Clone ERIC and install other dependencies
git clone https://github.com/RLE-Foundation/ERIC.git
cd ERIC
pip install -r requirements.txt

# 4. Install Flash Attention (performance critical)
pip install flash-attn==2.5.5 --no-build-isolation
```

### Flash Attention Installation Issues

If Flash Attention installation fails due to CUDA compilation issues, use this alternative method:

```bash
# Alternative: Download pre-compiled wheel
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install the downloaded wheel
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**Note**: This wheel is for:
- CUDA 12.2 (compatible with CUDA 12.1+)
- PyTorch 2.2
- Python 3.10
- Linux x86_64

### Verification

```python
import torch
import numpy as np
from prismatic.vla.action_tokenizer import ActionTokenizer
from libero.libero import benchmark

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✓ ERIC components loaded successfully")
```


# Algorithms Implemented

# Benchmark

# Cite Us

# Acknowledgements
