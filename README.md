# Unsloth Fine-Tuning – LLaMA-3.2-3B

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2×%20Faster-brightgreen)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Google Colab notebook for fine-tuning **LLaMA-3.2-3B-Instruct** using **Unsloth** with **LoRA + 4-bit quantization**.

The model is trained on the **ServiceNow-AI/R1-Distill-SFT** dataset to improve reflective, step-by-step reasoning.

</div>

---

## 🚀 Setup (Google Colab)

### Installation

```bash
!pip install -q unsloth
!pip install -q --force-reinstall --no-cache-dir --no-deps \
  git+https://github.com/unslothai/unsloth.git
```

> **Note:** Torch/torchaudio warnings in Colab are expected and safe to ignore.

---

## 🤖 Model Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | `unsloth/Llama-3.2-3B-Instruct` |
| **Quantization** | 4-bit |
| **Max Sequence Length** | 2048 |
| **Fine-tuning Method** | LoRA (PEFT) |

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | `ServiceNow-AI/R1-Distill-SFT` |
| **Split** | `v0 / train` |
| **Size** | ~171k samples |
| **Fields Used** | `problem`, `reannotated_assistant_content`, `solution` |

---

## 🎯 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 2 |
| **Gradient Accumulation** | 4 |
| **Effective Batch Size** | 8 |
| **Training Steps** | 60 |
| **Optimizer** | `adamw_8bit` |
| **Trainable Parameters** | ~0.75% |

---

## 💾 Output

After training, the model and tokenizer are saved locally:

```python
model.save_pretrained("nitin-001-3B")
tokenizer.save_pretrained("nitin-001-3B")
```

The fine-tuned model will be saved in the `nitin-001-3B` directory.

---

## 🔧 Notebook Rendering Fix (GitHub)

If GitHub shows **"Invalid Notebook"** or **widgets error**, clean metadata before uploading:

```python
import json

# Load the notebook
with open("Unsloth_FineTune.ipynb", "r") as f:
    nb = json.load(f)

# Remove problematic widgets metadata
nb["metadata"].pop("widgets", None)

# Save cleaned notebook
with open("notebook_clean.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
```

Then upload `notebook_clean.ipynb` to GitHub.

---

## 📝 Notes

- ✅ Runs on **Tesla T4** (15 GB VRAM)
- ✅ Unsloth enables **~2× faster fine-tuning**
- ✅ Attention mask warning during inference is harmless
- ✅ 4-bit quantization significantly reduces memory usage
- ✅ LoRA reduces trainable parameters to ~0.75% of total

---

## 📄 License

| Component | License |
|-----------|---------|
| **Base Model** | Meta LLaMA License |
| **Code** | MIT |
| **Dataset** | ServiceNow-AI terms |

---

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - For faster and memory-efficient fine-tuning
- [Meta AI](https://ai.meta.com/) - For the LLaMA-3.2 model
- [ServiceNow-AI](https://huggingface.co/ServiceNow-AI) - For the R1-Distill-SFT dataset

---

## 🔗 Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LLaMA-3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [R1-Distill-SFT Dataset](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
