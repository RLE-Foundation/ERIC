# ERIC
Embodied Reinforcement IntelligenCe (ERIC) Framework.

## Environment Setup

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

---
