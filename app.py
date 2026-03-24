import os
import time
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline


MODEL_PATH = "/mnt/d/AIModels/sd15"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading pipeline...")
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

print("Pipeline loaded.")


RESOLUTION_MAP = {
    "320 x 320 (faster)": (320, 320),
    "384 x 384 (recommended)": (384, 384),
    "448 x 448 (higher quality)": (448, 448),
}


def generate_image(prompt, negative_prompt, steps, guidance, resolution, seed):
    os.makedirs("outputs", exist_ok=True)

    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()

    if not prompt:
        return None, "Prompt cannot be empty."

    height, width = RESOLUTION_MAP[resolution]

    use_seed = int(seed)
    generator = None
    if use_seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(use_seed)

    start_time = time.time()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    elapsed = time.time() - start_time

    filename = f"sd_{int(time.time())}.png"
    output_path = os.path.join("outputs", filename)
    image.save(output_path)

    info = (
        f"Generation finished.\n"
        f"Steps: {steps}\n"
        f"Guidance: {guidance}\n"
        f"Resolution: {width}x{height}\n"
        f"Seed: {seed}\n"
        f"Time: {elapsed:.2f}s\n"
        f"Saved to: {output_path}"
    )

    return image, info


with gr.Blocks(title="Stable Diffusion Text-to-Image Demo") as demo:
    gr.Markdown(
        """
# 🎨 Stable Diffusion Text-to-Image Demo

本地 Stable Diffusion v1.5 文生图 Demo。  
推荐参数：**Steps = 30**，**Guidance = 10**，**384x384**。
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

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            output_info = gr.Textbox(label="Generation Info", lines=8)

    gr.Examples(
        examples=[
            [
                "a cute corgi wearing sunglasses, highly detailed, studio lighting",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
            ],
            [
                "a futuristic city at night, neon lights, cinematic, highly detailed",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
            ],
            [
                "an oil painting of a mountain village at sunset, warm colors, detailed",
                "blurry, low quality, distorted, ugly",
                30,
                10.0,
                "384 x 384 (recommended)",
                -1,
            ],
        ],
        inputs=[prompt, negative_prompt, steps, guidance, resolution, seed],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, resolution, seed],
        outputs=[output_image, output_info],
    )

if __name__ == "__main__":
    demo.launch()