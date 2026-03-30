import os
import time

_local_no_proxy = "localhost,127.0.0.1,::1"
for _key in ("NO_PROXY", "no_proxy"):
    _current = os.environ.get(_key)
    if _current:
        if "localhost" not in _current and "127.0.0.1" not in _current and "::1" not in _current:
            os.environ[_key] = f"{_current},{_local_no_proxy}"
    else:
        os.environ[_key] = _local_no_proxy

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline


MODEL_PATH = "/mnt/d/AIModels/sd15"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading base pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16",
    local_files_only=True,
)

if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
else:
    pipe = pipe.to("cpu")

print("Base pipeline loaded.")


RESOLUTION_MAP = {
    "320 x 320 (faster)": (320, 320),
    "384 x 384 (recommended)": (384, 384),
    "448 x 448 (higher quality)": (448, 448),
}

DEFAULT_LORA_DIR = "models/lora/archi_watercolor"
DEFAULT_LORA_WEIGHT = "[Lora][SD1.5]archi_watercolor-v1.safetensors"


def _pipe_kwargs(negative_prompt, steps, guidance, height, width, generator):
    return dict(
        prompt=None,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=height,
        width=width,
        generator=generator,
    )


def _generate_one(prompt, negative_prompt, steps, guidance, height, width, seed, lora_dir, lora_weight_name, lora_scale):
    generator = None
    if int(seed) >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    lora_status = "No LoRA loaded."
    if lora_dir and lora_weight_name:
        pipe.load_lora_weights(
            lora_dir,
            weight_name=lora_weight_name,
            local_files_only=True,
        )
        lora_status = f"Loaded LoRA: {os.path.join(lora_dir, lora_weight_name)}"

    start_time = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=height,
        width=width,
        generator=generator,
        cross_attention_kwargs={"scale": float(lora_scale)},
    ).images[0]
    elapsed = time.time() - start_time
    return image, elapsed, lora_status


def generate_image(prompt, negative_prompt, steps, guidance, resolution, seed, lora_dir, lora_weight_name, lora_scale):
    os.makedirs("outputs", exist_ok=True)

    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    lora_dir = lora_dir.strip()
    lora_weight_name = lora_weight_name.strip()

    if not prompt:
        return None, "Prompt cannot be empty."

    height, width = RESOLUTION_MAP[resolution]
    image, elapsed, lora_status = _generate_one(
        prompt, negative_prompt, steps, guidance, height, width, seed, lora_dir, lora_weight_name, lora_scale
    )

    filename = f"sd_{int(time.time())}.png"
    output_path = os.path.join("outputs", filename)
    image.save(output_path)

    info = (
        f"Generation finished.\n"
        f"Steps: {steps}\n"
        f"Guidance: {guidance}\n"
        f"Resolution: {width}x{height}\n"
        f"Seed: {seed}\n"
        f"LoRA scale: {lora_scale}\n"
        f"{lora_status}\n"
        f"Time: {elapsed:.2f}s\n"
        f"Saved to: {output_path}"
    )

    return image, info


def compare_images(prompt, negative_prompt, steps, guidance, resolution, seed, lora_dir, lora_weight_name, lora_scale):
    os.makedirs("outputs", exist_ok=True)

    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()

    if not prompt:
        return None, None, "Prompt cannot be empty."

    height, width = RESOLUTION_MAP[resolution]

    try:
        base_image, base_time, _ = _generate_one(
            prompt, negative_prompt, steps, guidance, height, width, seed, "", "", lora_scale
        )
        lora_image, lora_time, lora_status = _generate_one(
            prompt, negative_prompt, steps, guidance, height, width, seed, lora_dir, lora_weight_name, lora_scale
        )
    except Exception as e:
        return None, None, f"Comparison failed.\nError: {str(e)}"

    base_path = os.path.join("outputs", f"base_{int(time.time())}.png")
    lora_path = os.path.join("outputs", f"lora_{int(time.time())}.png")
    base_image.save(base_path)
    lora_image.save(lora_path)

    info = (
        f"Baseline time: {base_time:.2f}s\n"
        f"LoRA time: {lora_time:.2f}s\n"
        f"{lora_status}\n"
        f"Baseline saved to: {base_path}\n"
        f"LoRA saved to: {lora_path}"
    )
    return base_image, lora_image, info
with gr.Blocks(title="Stable Diffusion Text-to-Image Demo with LoRA Interface") as demo:
    gr.Markdown(
        """
# 🎨 Stable Diffusion Text-to-Image Demo

本地 Stable Diffusion v1.5 文生图 Demo。  
支持基础生成，并预留了 **LoRA 接口** 方便后续扩展。

推荐基础参数：**Steps = 30**，**Guidance = 10**，**384x384**。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="e.g. a cute corgi wearing sunglasses, highly detailed, studio lighting",
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                value="blurry, low quality, distorted, ugly",
            )

            steps = gr.Slider(
                minimum=10,
                maximum=40,
                value=30,
                step=5,
                label="Inference Steps",
            )

            guidance = gr.Slider(
                minimum=1.0,
                maximum=12.0,
                value=10.0,
                step=0.5,
                label="Guidance Scale",
            )

            resolution = gr.Dropdown(
                choices=list(RESOLUTION_MAP.keys()),
                value="384 x 384 (recommended)",
                label="Resolution",
            )

            seed = gr.Number(
                value=-1,
                precision=0,
                label="Seed (-1 means random)",
            )

            gr.Markdown("## Optional LoRA Settings")

            lora_dir = gr.Textbox(
                label="LoRA Directory",
                value=DEFAULT_LORA_DIR,
                placeholder="e.g. models/lora/archi_watercolor",
            )

            lora_weight_name = gr.Textbox(
                label="LoRA Weight File Name",
                value=DEFAULT_LORA_WEIGHT,
                placeholder="e.g. [Lora][SD1.5]archi_watercolor-v1.safetensors",
            )

            lora_scale = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.8,
                step=0.1,
                label="LoRA Scale",
            )

            generate_btn = gr.Button("Generate", variant="primary")
            compare_btn = gr.Button("Compare Base vs LoRA")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            output_base_image = gr.Image(label="Baseline Image")
            output_lora_image = gr.Image(label="LoRA Image")
            output_info = gr.Textbox(label="Generation Info", lines=10)

    gr.Examples(
        examples=[
            [
                "a cute corgi wearing sunglasses, highly detailed, studio lighting",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
                "",
                "",
                0.8,
            ],
            [
                "a futuristic city at night, neon lights, cinematic, highly detailed",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
                "",
                "",
                0.8,
            ],
            [
                "an oil painting of a mountain village at sunset, warm colors, detailed",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
                "",
                "",
                0.8,
            ],
        ],
        inputs=[prompt, negative_prompt, steps, guidance, resolution, seed, lora_dir, lora_weight_name, lora_scale],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, resolution, seed, lora_dir, lora_weight_name, lora_scale],
        outputs=[output_image, output_info],
    )

    compare_btn.click(
        fn=compare_images,
        inputs=[prompt, negative_prompt, steps, guidance, resolution, seed, lora_dir, lora_weight_name, lora_scale],
        outputs=[output_base_image, output_lora_image, output_info],
    )

if __name__ == "__main__":
    print("Browser URL: http://127.0.0.1:7860", flush=True)
    demo.launch(server_name="0.0.0.0", show_error=True)
