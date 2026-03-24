# 🎨 Stable Diffusion Gradio Demo

A local text-to-image demo built with **Stable Diffusion v1.5 + Gradio**.

---

## Features

- Text-to-image generation
- Adjustable inference steps
- Adjustable guidance scale
- Multiple resolution options
- Seed control for reproducibility
- Local model loading (offline inference)

---

## Tech Stack

- PyTorch
- Diffusers
- Transformers
- Gradio

---

## Model

- Stable Diffusion v1.5
- Loaded from local directory
- FP16 inference

---

## Recommended Settings

- Steps: 30
- Guidance Scale: 10
- Resolution: 384x384

---

## Usage

```bash
python app.py

