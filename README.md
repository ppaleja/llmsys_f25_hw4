# MiniTorch: GPU-Accelerated Transformer Framework

A high-performance deep learning framework implementing GPU-accelerated transformer models from scratch. This project demonstrates the complete pipeline from low-level CUDA kernels to transformer architecture, culminating in a machine translation system with optimized attention and normalization operations.

## Overview

This repository represents the culmination of building a complete deep learning framework across four major components:

1. **CUDA Programming (Assignment 1)**: Custom CUDA kernels for fundamental tensor operations
2. **MiniTorch Framework (Assignment 2)**: Automatic differentiation and neural network modules
3. **Transformer Architecture (Assignment 3)**: Full decoder-only transformer (GPT-2 style) implementation
4. **CUDA Acceleration (Assignment 4)**: Fused softmax and layernorm kernels for optimized training

The framework is capable of training transformer models on real tasks like machine translation (IWSLT14 German-English dataset) with significant speedups from custom CUDA optimizations.

## Features

### Core Tensor Operations (Assignment 1)
- **Map Operations**: Element-wise unary operations (ReLU, sigmoid, log, exp, etc.)
- **Zip Operations**: Element-wise binary operations (add, multiply, compare, etc.)
- **Reduce Operations**: Dimension-wise aggregation (sum, max, etc.)
- **Matrix Multiplication**: Optimized GPU-accelerated matrix multiplication with shared memory

### Automatic Differentiation (Assignment 2)
- **Computation Graph**: Automatic construction of computational graphs
- **Backpropagation**: Efficient gradient computation via reverse-mode autodiff
- **Neural Network Modules**: Modular building blocks (Linear, Embedding, Dropout, etc.)

### Transformer Architecture (Assignment 3)
- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Feature normalization for stable training
- **Feed-Forward Networks**: Position-wise fully connected layers with GELU activation
- **Decoder-Only Model**: GPT-2 style autoregressive transformer
- **Machine Translation**: Training on IWSLT14 German-English dataset

### Optimized CUDA Kernels (Assignment 4)
- **Fused Softmax**: Forward and backward kernels with ~6.5× speedup (forward)
  - Warp-level primitives for sequences <32
  - Block-level reduction for longer sequences
  - Attention mask support
- **Fused LayerNorm**: Forward and backward kernels with ~15.8× speedup (forward)
  - Concurrent computation of mean and variance
  - float4 vectorization for memory efficiency
  - Efficient gradient computation
- **End-to-End Speedup**: ~1.1× overall training speedup when integrated into transformer

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: CUDA Toolkit 12.0+ with NVCC compiler
- **GPU**: NVIDIA GPU with compute capability 7.5+ (recommended)
- **RAM**: 8GB+ recommended

**GPU Access Options**:
- Google Colab (free, T4 GPU available)
- AWS with GPU instances
- PSC (Pittsburgh Supercomputing Center)
- Local machine with NVIDIA GPU

## Installation

### 1. Set up a Virtual Environment

Using conda (recommended):
```bash
conda create -n minitorch-cuda python=3.9.16
conda activate minitorch-cuda
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Load CUDA Module (if using PSC or similar systems)

```bash
module load anaconda3/2024.10-1
module load cuda/12.4
```

### 3. Install PyTorch with CUDA Support

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Clone and Install MiniTorch

```bash
git clone https://github.com/ppaleja/llmsys_f25_hw4.git
cd llmsys_f25_hw4
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install -Ue .
```

### 5. Verify Installation

```bash
python -c "import minitorch; print('Success: minitorch is installed correctly');"
```

### 6. Compile CUDA Kernels

```bash
bash compile_cuda.sh
```

Or manually:
```bash
mkdir -p minitorch/cuda_kernels
nvcc -O2 -arch=sm_75 -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -O2 -arch=sm_75 -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
nvcc -O2 -arch=sm_75 -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
```

**Note**: Adjust `-arch=sm_75` based on your GPU's compute capability. Common values:
- `sm_75`: RTX 20 series, T4
- `sm_80`: A100
- `sm_86`: RTX 30 series
- `sm_89`: RTX 40 series

## Project Structure

```
llmsys_f25_hw4/
├── minitorch/                    # Core MiniTorch framework
│   ├── autodiff.py               # Automatic differentiation (topological sort, backprop)
│   ├── cuda_kernel_ops.py        # CUDA kernel bindings
│   ├── cuda_ops.py               # CUDA tensor operations
│   ├── tensor.py                 # Tensor data structure
│   ├── tensor_ops.py             # Tensor operation interfaces
│   ├── tensor_functions.py       # High-level tensor functions
│   ├── operators.py              # Mathematical operators
│   ├── module.py                 # Base Module and Parameter classes
│   ├── modules_basic.py          # Basic modules (Linear, Embedding, Dropout, LayerNorm1d)
│   ├── modules_transfomer.py     # Transformer modules (MultiHeadAttention, TransformerLayer, DecoderLM)
│   ├── nn.py                     # Neural network utilities (softmax, GELU, etc.)
│   └── optim.py                  # Optimizers (SGD, Adam)
│
├── src/                          # CUDA kernel implementations
│   ├── combine.cu                # Map, zip, reduce, matmul kernels (Assignment 1)
│   ├── softmax_kernel.cu         # Fused softmax kernels (Assignment 4)
│   ├── layernorm_kernel.cu       # Fused layernorm kernels (Assignment 4)
│   └── includes/                 # CUDA helper headers
│
├── project/                      # Training scripts
│   └── run_machine_translation.py # Machine translation training
│
├── tests/                        # Unit tests
│   ├── test_autodiff.py          # Autodiff tests
│   ├── test_tensor_general.py    # Tensor operation tests
│   ├── test_modules_transformer.py # Transformer module tests
│   └── ...
│
├── kernel_tests/                 # CUDA kernel benchmarks
│   ├── test_softmax_fw.py        # Softmax forward benchmarks
│   ├── test_softmax_bw.py        # Softmax backward benchmarks
│   ├── test_layernorm_fw.py      # LayerNorm forward benchmarks
│   └── test_layernorm_bw.py      # LayerNorm backward benchmarks
│
├── compile_cuda.sh               # CUDA compilation script
├── requirements.txt              # Python dependencies
├── requirements.extra.txt        # Additional dependencies
└── Project_Demo.ipynb            # Interactive demo notebook
```

## Quick Start

For a comprehensive walkthrough of all features with working examples, see the [Project_Demo.ipynb](Project_Demo.ipynb) notebook.

### Basic Usage

```python
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

# Create CUDA backend
backend = minitorch.TensorBackend(CudaKernelOps)

# Create tensors
x = minitorch.tensor([[1.0, 2.0], [3.0, 4.0]], backend=backend)
y = minitorch.tensor([[5.0, 6.0], [7.0, 8.0]], backend=backend)

# Perform operations
z = x + y  # Element-wise addition
w = x @ y  # Matrix multiplication
```

### Transformer Training

Train a transformer model on machine translation:

```bash
# Without fused kernels
python project/run_machine_translation.py --use-fused-kernel False

# With fused kernels (faster)
python project/run_machine_translation.py --use-fused-kernel True
```

## Module Descriptions

### Assignment 1: CUDA Tensor Operations

**Implementation**: `src/combine.cu`, `minitorch/cuda_kernel_ops.py`

- **Map Kernel**: Applies unary operations element-wise. Each thread processes one output element with stride-based indexing for multidimensional tensors.
- **Zip Kernel**: Applies binary operations element-wise on two tensors. Supports broadcasting.
- **Reduce Kernel**: Performs reduction along specified dimensions. Uses shared memory for efficient block-level reduction.
- **Matrix Multiply Kernel**: Optimized using shared memory tiling (TILE=32) to minimize global memory access.

### Assignment 2: Automatic Differentiation

**Implementation**: `minitorch/autodiff.py`

- **Topological Sort**: Computes reverse topological order of computation graph using depth-first search.
- **Backpropagation**: Traverses computation graph in topological order, applying chain rule to compute gradients.
- **Scalar Operations**: Foundation for tensor autodiff with support for basic mathematical operations.

### Assignment 3: Transformer Architecture

**Implementation**: `minitorch/modules_transfomer.py`, `minitorch/modules_basic.py`

- **Embedding**: Maps word indices to continuous vector representations
- **MultiHeadAttention**: Implements scaled dot-product attention with multiple heads
  - Query, Key, Value projections
  - Attention score computation with optional causal masking
  - Output projection
- **TransformerLayer**: Combines multi-head attention with position-wise feed-forward network
  - Pre-layer normalization
  - Residual connections
  - GELU activation
- **DecoderLM**: Full decoder-only transformer for language modeling
  - Token + positional embeddings
  - Stacked transformer layers
  - Output projection to vocabulary

### Assignment 4: Optimized CUDA Kernels

**Implementation**: `src/softmax_kernel.cu`, `src/layernorm_kernel.cu`

#### Fused Softmax
- **Forward**: 
  - Two-pass algorithm: (1) find max, (2) compute exp and sum, (3) normalize
  - `ker_attn_softmax_lt32`: Warp-level reduction for short sequences (<32)
  - `ker_attn_softmax`: Block-level reduction using CUB library for longer sequences
  - Integrated attention mask support
  - **Speedup**: ~6.5× vs. PyTorch
  
- **Backward**:
  - Computes gradient using: `grad_input = softmax_output * (grad_output - sum(softmax_output * grad_output))`
  - Template-based kernel tuning for different sequence lengths
  - **Speedup**: ~0.5× vs. PyTorch (baseline)

#### Fused LayerNorm
- **Forward**:
  - Single-pass computation of mean and variance using Welford's algorithm variant
  - float4 vectorization for memory coalescing
  - Concurrent computation of E[x] and E[x²] to derive variance
  - **Speedup**: ~15.8× vs. PyTorch
  
- **Backward**:
  - Separate kernels for input gradient and parameter gradients
  - `ker_ln_bw_dinp`: Computes input gradients using batch statistics
  - `ker_ln_bw_dgamma_dbetta`: Computes gamma and beta gradients using shared memory
  - Warp shuffle operations for efficient reduction
  - **Speedup**: ~3.7× vs. PyTorch

## Performance Benchmarks

### Kernel-Level Speedups

| Kernel | Operation | Speedup vs PyTorch |
|--------|-----------|-------------------|
| Softmax | Forward | ~6.5× |
| Softmax | Backward | ~0.5× |
| LayerNorm | Forward | ~15.8× |
| LayerNorm | Backward | ~3.7× |

### End-to-End Training Speedup

Training transformer on IWSLT14 German-English:
- **Without fused kernels**: Baseline performance
- **With fused kernels**: ~1.1× faster
- **Overall speedup**: ~1.1×

*Note: The modest end-to-end speedup (per Amdahl's law) is due to softmax and layernorm representing a small fraction of total computation time.*

## Testing

### Run All Tests

```bash
# Unit tests
pytest tests/

# CUDA kernel benchmarks
python kernel_tests/test_softmax_fw.py
python kernel_tests/test_softmax_bw.py
python kernel_tests/test_layernorm_fw.py
python kernel_tests/test_layernorm_bw.py
```

### Specific Test Categories

```bash
# Autodiff tests
pytest tests/test_autodiff.py

# Tensor operation tests
pytest tests/test_tensor_general.py

# Transformer module tests
pytest tests/test_modules_transformer.py
```

## Development Notes

### Assignment Evolution

This repository (Assignment 4) builds upon code from previous assignments:
- **Assignment 1**: CUDA kernels in `src/combine.cu`
- **Assignment 2**: Autodiff framework in `minitorch/autodiff.py`
- **Assignment 3**: Transformer modules in `minitorch/modules_transfomer.py`, `minitorch/modules_basic.py`, `minitorch/nn.py`
- **Assignment 4**: Fused kernels in `src/softmax_kernel.cu`, `src/layernorm_kernel.cu`

### Key Implementation Details

1. **Stride-based Indexing**: Enables efficient handling of transposed and strided tensors without data copying
2. **Shared Memory Optimization**: Used in matrix multiply and reduction operations to minimize global memory bandwidth
3. **Warp-level Primitives**: Leveraged in softmax and layernorm for efficient intra-warp communication
4. **float4 Vectorization**: Quadruples memory throughput in layernorm kernels
5. **CUB Library**: Utilized for optimized block-level reductions in larger softmax computations

## References

- **LightSeq**: [LightSeq: A High Performance Inference Library for Transformers](https://arxiv.org/pdf/2010.13887)
- **LightSeq2**: [LightSeq2: Accelerated Training for Transformer-based Models on GPUs](https://arxiv.org/pdf/2110.05722)
- **Attention Is All You Need**: [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- **MiniTorch**: Inspired by the MiniTorch educational framework

## License

This project is part of CMU 11-868 LLM Systems (Fall 2025) coursework.

## Acknowledgments

- Course staff and instructors of CMU 11-868
- MiniTorch educational framework
- LightSeq and LightSeq2 papers for optimization techniques
