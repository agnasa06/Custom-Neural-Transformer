
# Custom Neural Transformer 🧠

A transformer-based language model built from scratch using Python and PyTorch, with a focus on understanding neural network architectures, training algorithms, and computational efficiency.

This project began as an experiment in implementing neural networks manually and evolved into a custom transformer with GPU-accelerated training, adaptive optimization, and text generation.

## ⚙️ Architecture

The model uses a decoder-only transformer architecture for character-level language modeling.

**Current configuration:**

| Component | Configuration |
|-----------|---------------|
| Architecture | Decoder-only Transformer |
| Layers | 12 |
| Embedding Dimensions | 768 |
| Attention Heads | 6 |
| Context Length | 384 characters |
| Vocabulary | 221 character tokens |
| Positional Encoding | Rotary Positional Embeddings (RoPE) |
| Attention | Scaled Dot-Product Attention |
| Framework | PyTorch |

The model incorporates multi-head self-attention, feedforward networks, residual connections, and causal masking to predict subsequent characters.

## 🔬 Training & Optimization

A major focus of this project is experimenting with how neural networks learn and how training can be made more efficient.

Key implementations include:

- **Custom Optimizer:** Developed an adaptive optimization approach incorporating gradient momentum, historical statistics, and parameter-specific scaling.
- **GPU Acceleration:** Experimented with PyTorch compilation, mixed-precision training, and GPU-accelerated operations.
- **Adaptive Updates:** Implemented separate optimization statistics for query, key, value, and feedforward projections.
- **Training Calibration:** Added an initialization phase to establish reference statistics for adaptive parameter updates.
- **Checkpointing:** Implemented model saving and restoration, including optimizer states and training progress.

## 📚 Dataset & Tokenization

The model uses character-level tokenization, predicting the next character based on preceding context.

The current implementation supports:

- FineWeb-Edu training data
- A custom 221-character vocabulary
- Unicode normalization and text preprocessing
- Memory-mapped datasets for efficient data loading

Earlier versions also experimented with WikiText datasets and smaller character vocabularies.

## 🖥️ Implementation

The repository contains the following primary files:

- **`VersionOfficial.py`**: Main transformer architecture, training loop, and custom optimization logic.
- **`RunningAI.py`**: Model inference and text generation.
- **`DataTokenization.py`**: Dataset preprocessing and character-level tokenization.
- **`VersionOfficial_GitHub_Description.md`**: Detailed development notes and implementation changes.

## 🚀 Development

The project has evolved through several stages:

1. Implementing neural network operations and training logic manually.
2. Transitioning to tensor-based computation with PyTorch.
3. Building a transformer architecture with attention and feedforward layers.
4. Experimenting with custom optimization and gradient statistics.
5. Improving GPU performance through compilation and mixed-precision operations.
6. Expanding the architecture, dataset, and inference capabilities.

## 🎯 Goals

This project is an ongoing exploration of neural network fundamentals, particularly the relationship between mathematical optimization, model architecture, and computational performance.

Future experiments will focus on improving training efficiency, evaluating optimization techniques, and exploring larger or more capable architectures.

---

**Built with:** Python • PyTorch • CUDA • TorchAO
