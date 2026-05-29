<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F97316,50:EF4444,100:DC2626&height=220&section=header&text=Transformer&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Built%20from%20Scratch%20in%20PyTorch&descSize=18&descAlignY=55&descAlign=50" width="100%"/>

<!-- Typing SVG -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F97316&center=true&vCenter=true&width=700&lines=Attention+Is+All+You+Need+(2017);Multi-Head+Self-Attention;Positional+Encoding+%7C+Encoder-Decoder;No+nn.Transformer+%E2%80%94+Pure+PyTorch" alt="Typing SVG" /></a>

<br/>

<!-- Badges -->
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />

<br/><br/>

<!-- Paper Badge -->
<a href="https://arxiv.org/abs/1706.03762"><img src="https://img.shields.io/badge/arXiv-1706.03762-b31b1b?style=flat-square&logo=arxiv&logoColor=white" /></a>
<img src="https://img.shields.io/badge/Vaswani%20et%20al.-NeurIPS%202017-F97316?style=flat-square&labelColor=1a1a2e" />

</div>

---

<div align="center">
<h2>🎯 What This Is</h2>
<p><i>A complete from-scratch PyTorch implementation of the original Transformer architecture.<br/>Every component — multi-head attention, positional encoding, encoder, decoder — written by hand without <code>nn.Transformer</code>.</i></p>
</div>

---

<div align="center">
<h2>🏗️ Architecture</h2>
</div>

```mermaid
graph TD
    subgraph Encoder["🔵 ENCODER (N layers)"]
        A["Source Tokens"] --> AE["Embedding + Positional Encoding"]
        AE --> SA["Multi-Head Self-Attention"]
        SA --> AN1["Add & Norm"]
        AN1 --> FF1["Feed-Forward Network"]
        FF1 --> AN2["Add & Norm"]
    end
    
    subgraph Decoder["🔴 DECODER (N layers)"]
        B["Target Tokens"] --> BE["Embedding + Positional Encoding"]
        BE --> MSA["Masked Self-Attention"]
        MSA --> AN3["Add & Norm"]
        AN3 --> CA["Cross-Attention"]
        AN2 -->|"K, V"| CA
        CA --> AN4["Add & Norm"]
        AN4 --> FF2["Feed-Forward Network"]
        FF2 --> AN5["Add & Norm"]
    end
    
    AN5 --> LIN["Linear + Softmax"]
    LIN --> OUT["📤 Output Tokens"]
    
    style A fill:#3B82F6,stroke:#3B82F6,color:#fff
    style AE fill:#60A5FA,stroke:#60A5FA,color:#fff
    style SA fill:#2563EB,stroke:#2563EB,color:#fff
    style AN1 fill:#1D4ED8,stroke:#1D4ED8,color:#fff
    style FF1 fill:#1E40AF,stroke:#1E40AF,color:#fff
    style AN2 fill:#1D4ED8,stroke:#1D4ED8,color:#fff
    style B fill:#EF4444,stroke:#EF4444,color:#fff
    style BE fill:#F87171,stroke:#F87171,color:#fff
    style MSA fill:#DC2626,stroke:#DC2626,color:#fff
    style AN3 fill:#B91C1C,stroke:#B91C1C,color:#fff
    style CA fill:#F97316,stroke:#F97316,color:#fff
    style AN4 fill:#B91C1C,stroke:#B91C1C,color:#fff
    style FF2 fill:#991B1B,stroke:#991B1B,color:#fff
    style AN5 fill:#B91C1C,stroke:#B91C1C,color:#fff
    style LIN fill:#10B981,stroke:#10B981,color:#fff
    style OUT fill:#059669,stroke:#059669,color:#fff
```

---

<div align="center">
<h2>🧩 Components Implemented</h2>
</div>

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/🔍-Attention-F97316?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Scaled Dot-Product</b><br/>
Q·K^T / √d_k<br/>
+ Softmax + V
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/🧠-Multi_Head-EF4444?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Parallel Heads</b><br/>
h independent attention<br/>
functions concatenated
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/📍-Position-DC2626?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Sinusoidal Encoding</b><br/>
sin/cos position<br/>
embeddings
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/📐-Masking-B91C1C?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Causal Mask</b><br/>
Future token<br/>
prevention
</td>
</tr>
</table>

<details>
<summary><b>📋 Full Component List (click to expand)</b></summary>
<br/>

| Component | Description | Status |
|:----------|:------------|:------:|
| Scaled Dot-Product Attention | Core Q·K·V attention mechanism | ✅ |
| Multi-Head Attention | Parallel attention heads | ✅ |
| Positional Encoding | Sinusoidal position embeddings | ✅ |
| Encoder Block | Self-Attn → Add&Norm → FFN → Add&Norm | ✅ |
| Decoder Block | Masked Self-Attn → Cross-Attn → FFN | ✅ |
| Layer Normalization | Pre/post-norm variants | ✅ |
| Feed-Forward Network | 2-layer MLP + ReLU | ✅ |
| Encoder-Decoder Stack | Complete seq2seq architecture | ✅ |
| Training Loop | Full training pipeline | ✅ |

</details>

---

<div align="center">
<h2>🚀 Quick Start</h2>
</div>

```bash
git clone https://github.com/udayraj1238/Transformer_from_scratch_using_pytorch.git
cd Transformer_from_scratch_using_pytorch
pip install torch torchvision
```

---

<div align="center">
<h2>📚 Reference</h2>

<a href="https://arxiv.org/abs/1706.03762"><img src="https://img.shields.io/badge/📄_Attention_Is_All_You_Need-Vaswani_et_al.,_NeurIPS_2017-b31b1b?style=for-the-badge" /></a>

<br/><br/>

<h2>🤝 Contact</h2>
<a href="https://www.linkedin.com/in/uday6002/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
<a href="https://udayraj1238.vercel.app"><img src="https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white" /></a>
<a href="mailto:rajuday6002@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" /></a>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F97316,50:EF4444,100:DC2626&height=120&section=footer" width="100%"/>
