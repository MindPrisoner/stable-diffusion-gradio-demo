---
license: creativeml-openrail-m
base_model: runwayml/stable-diffusion-v1-5
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- lora
- minhwa
- korean-art
inference: true
---

# 🐯 Living Minhwa: Korean Folk Art LoRA

**Living Minhwa**는 한국 전통 민화(Minhwa), 특히 '까치호랑이'와 '화조도'의 화풍을 학습한 Stable Diffusion LoRA 모델입니다.  
단순한 스타일 모방을 넘어, **Aspect Ratio Bucketing** 기술을 통해 족자(Scroll)와 병풍(Screen)의 구도까지 학습했습니다.

## 💡 Model Details
- **Model Type:** SD 1.5 LoRA
- **Rank/Alpha:** 32 / 16 (Optimized for fidelity)
- **Trigger Word:** `minhwa style`
- **Training Data:** 70 High-quality images from National Museum of Korea (e-Museum)

## 🎨 How to Use (Prompt Engineering)

이 모델은 **"Minhwa Cheat Code"** (특수 프롬프트)를 사용할 때 가장 성능이 좋습니다.

### Recommended Settings
- **LoRA Weight:** 0.8 ~ 1.0
- **Sampler:** DPM++ 2M Karras or Euler a
- **CFG Scale:** 7.0

### ✨ The "Cheat Code" Prompt
Use this suffix to enforce texture and flat perspective:
```text
, minhwa style, (masterpiece, best quality:1.4), (traditional korean ink painting:1.3), (hanji paper texture:1.3), rough brush strokes, flat perspective, vivid colors