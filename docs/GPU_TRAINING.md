# GPU Training Guide

This document describes how to use GPU acceleration for training and inference in ReUnity.

## Overview

ReUnity is designed to work in CPU-only environments like GitHub Codespaces, but can leverage GPU acceleration when available for improved performance in embedding generation and FAISS indexing.

## GPU Detection

ReUnity automatically detects GPU availability:

```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Running on CPU")
```

## FAISS GPU Support

### Installation

For GPU-accelerated FAISS:

```bash
# CPU-only (default)
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

### Usage

```python
from reunity.rag.indexer import FAISSIndexer

# GPU-accelerated indexer
indexer = FAISSIndexer(
    embedding_dim=384,
    index_type="flat",
    use_gpu=True,  # Enable GPU if available
)
```

### Performance Comparison

| Operation | CPU (1M vectors) | GPU (1M vectors) |
|-----------|------------------|------------------|
| Index build | ~60s | ~5s |
| Search (k=10) | ~100ms | ~5ms |
| Batch search (100 queries) | ~10s | ~500ms |

## Embedding Model Training

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model (automatically uses GPU if available)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Explicit device selection
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

### Fine-tuning for Domain

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=['grounding technique', '5-4-3-2-1 method'], label=0.9),
    InputExample(texts=['emotional regulation', 'managing feelings'], label=0.85),
    # Add more domain-specific pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='models/reunity-embeddings'
)
```

## Training Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPUs to use (e.g., "0,1") |
| `REUNITY_USE_GPU` | auto | Force GPU usage (true/false/auto) |
| `REUNITY_BATCH_SIZE` | 32 | Batch size for GPU operations |

### Memory Management

```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Cloud GPU Options

### Google Colab

```python
# Check GPU in Colab
!nvidia-smi

# Install dependencies
!pip install faiss-gpu sentence-transformers

# Clone and setup
!git clone https://github.com/ezernackchristopher97-cloud/ReUnity.git
%cd ReUnity
!pip install -e .
```

### AWS SageMaker

```yaml
# sagemaker-config.yaml
instance_type: ml.g4dn.xlarge
framework: pytorch
py_version: py310
```

### Azure ML

```python
from azureml.core import Workspace, Environment

env = Environment.from_pip_requirements(
    name='reunity-gpu',
    file_path='requirements-gpu.txt'
)
env.docker.base_image = 'mcr.microsoft.com/azureml/pytorch-1.10-cuda11.3-cudnn8-runtime'
```

## Benchmarking

### Running Benchmarks

```bash
# CPU benchmark
python scripts/benchmark.py --device cpu

# GPU benchmark
python scripts/benchmark.py --device cuda
```

### Expected Results

| Model | CPU (queries/sec) | GPU (queries/sec) |
|-------|-------------------|-------------------|
| all-MiniLM-L6-v2 | ~50 | ~500 |
| all-mpnet-base-v2 | ~20 | ~200 |
| Custom fine-tuned | ~30 | ~300 |

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
indexer = FAISSIndexer(embedding_dim=384, batch_size=16)

# Use gradient checkpointing for training
model.gradient_checkpointing_enable()
```

### CUDA Version Mismatch

```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Install matching versions
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Driver Issues

```bash
# Check driver
nvidia-smi

# Update driver (Ubuntu)
sudo apt update
sudo apt install nvidia-driver-525
```

## CPU Fallback

ReUnity automatically falls back to CPU when GPU is unavailable:

```python
# This works on both CPU and GPU
from reunity.rag.indexer import FAISSIndexer

indexer = FAISSIndexer(
    embedding_dim=384,
    use_gpu=True,  # Will use CPU if GPU unavailable
)

# Check what's being used
print(indexer.get_statistics())
# Output: {"has_faiss": true, "using_gpu": false, ...}
```

## Best Practices

1. **Start with CPU**: Develop and test on CPU first
2. **Profile before optimizing**: Identify actual bottlenecks
3. **Use mixed precision**: FP16 for faster training with minimal accuracy loss
4. **Batch operations**: Amortize GPU transfer overhead
5. **Monitor memory**: Prevent OOM errors with proper batch sizing

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.
