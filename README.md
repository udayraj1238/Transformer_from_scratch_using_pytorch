# 🔧 Transformer from Scratch — PyTorch Implementation

> Complete implementation of the **Transformer architecture** from the "Attention Is All You Need" paper, built entirely from scratch in PyTorch.

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
</p>

---

## 🎯 What This Is

A from-scratch PyTorch implementation of the original Transformer model introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). Every component — multi-head attention, positional encoding, encoder, decoder, layer normalization — is implemented by hand without relying on `nn.Transformer`.

---

## 🏗️ Components Implemented

| Component | Description |
|-----------|-------------|
| **Scaled Dot-Product Attention** | Core attention mechanism with query, key, value matrices |
| **Multi-Head Attention** | Parallel attention heads for diverse representation learning |
| **Positional Encoding** | Sinusoidal position embeddings for sequence order information |
| **Encoder Block** | Self-attention → Add & Norm → Feed-Forward → Add & Norm |
| **Decoder Block** | Masked self-attention → Cross-attention → Feed-Forward |
| **Layer Normalization** | Pre/post-norm variants for training stability |
| **Feed-Forward Network** | Two-layer MLP with ReLU activation |
| **Full Encoder-Decoder** | Complete seq2seq architecture with masking |

---

## 🏗️ Architecture

```
┌─────────────┐              ┌─────────────┐
│   Source     │              │   Target     │
│   Tokens     │              │   Tokens     │
└──────┬──────┘              └──────┬──────┘
       │                            │
       ▼                            ▼
┌──────────────┐             ┌──────────────┐
│  Embedding + │             │  Embedding + │
│  Positional  │             │  Positional  │
│  Encoding    │             │  Encoding    │
└──────┬───────┘             └──────┬───────┘
       │                            │
       ▼                            │
┌──────────────┐                    │
│              │                    │
│   ENCODER    │                    ▼
│   (N layers) │             ┌──────────────┐
│              │────────────▶│              │
│  Self-Attn   │   K, V      │   DECODER    │
│  Feed-Fwd    │             │   (N layers) │
│              │             │              │
└──────────────┘             │  Masked Attn │
                             │  Cross-Attn  │
                             │  Feed-Fwd    │
                             │              │
                             └──────┬───────┘
                                    │
                                    ▼
                             ┌──────────────┐
                             │   Linear +   │
                             │   Softmax    │
                             │   Output     │
                             └──────────────┘
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/udayraj1238/Transformer_from_scratch_using_pytorch.git
cd Transformer_from_scratch_using_pytorch
pip install -r requirements.txt
```

---

## 📚 Reference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contact

**Uday Raj** — [LinkedIn](https://www.linkedin.com/in/uday6002/) · [Portfolio](https://udayraj1238.vercel.app) · [Email](mailto:rajuday6002@gmail.com)
