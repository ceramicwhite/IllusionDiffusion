import torch
import os
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "XpucT/Deliberate",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def inference(
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 1.0,
    strength: float = 0.8,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")
    
    input_image = input_image.resize((512, 512))
    
    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()
    
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        control_image=input_image,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        strength=float(strength),
        num_inference_steps=40,
    )
    return out.images[0]

with gr.Blocks() as app:
    gr.Markdown(
        '''
        # Illusion Diffusion 🌀
        ## Generate beautiful illusion art with SD 1.5.
        **[Follow me on Twitter](https://twitter.com/angrypenguinPNG)**
        '''
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Illusion", type="pil")
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="ugly, disfigured, low quality, blurry, nsfw")
            with gr.Accordion(label="Advanced Options", open=False):
                controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=1.1, label="Controlnet Conditioning Scale")
                strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Strength")
                guidance_scale = gr.Slider(minimum=0.0, maximum=50.0, step=0.25, value=7.5, label="Guidance Scale")
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="DPM++ Karras SDE")
                seed = gr.Slider(minimum=-1, maximum=9999999999, step=1, value=2313123, label="Seed", randomize=True)
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Illusion Diffusion Output")
            
    run_btn.click(
        inference,
        inputs=[input_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, strength, seed, sampler],
        outputs=[result_image]
    )

app.queue(concurrency_count=4, max_size=20)

if __name__ == "__main__":
    app.launch(debug=True)
