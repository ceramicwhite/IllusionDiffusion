import torch
import os
import gradio as gr
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

# Initialize both pipelines
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
#init_pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)
main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
#model_id = "stabilityai/sd-x2-latent-upscaler"
#upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#upscaler.to("cuda")


# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img

# Inference function
def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
    progress = gr.Progress(track_tqdm=True)
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")
    
    # Generate the initial image
    #init_image = init_pipe(prompt).images[0]

    # Rest of your existing code
    control_image = center_crop_resize(control_image)
    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        #control_image=control_image,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        #strength=strength,
        num_inference_steps=30,
        #output_type="latent"
    ).images[0]
    
    return out

with gr.Blocks() as app:
    gr.Markdown(
        '''
        <center><h1>Illusion Diffusion 🌀</h1></span>  
        <span font-size:16px;">Generate stunning illusion artwork with Stable Diffusion</span>  
        </center>
 
        A space by AP [Follow me on Twitter](https://twitter.com/angrypenguinPNG)

        This project works by using [Monster Labs QR Control Net](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster).
        Given a prompt and your pattern, we use a QR code conditioned controlnet to create a stunning illusion! Credit to: MrUgleh (https://twitter.com/MrUgleh) for discovering the workflow :)

        '''
    )
    
    with gr.Row():
        with gr.Column():
            control_image = gr.Image(label="Input Illusion", type="pil")
            controlnet_conditioning_scale = gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.8, label="Illusion strength", info="ControlNet conditioning scale")
            gr.Examples(examples=["checkers.png", "pattern.png", "spiral.jpeg"], inputs=control_image)
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality")
            with gr.Accordion(label="Advanced Options", open=False):
                #strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Strength")
                guidance_scale = gr.Slider(minimum=0.0, maximum=50.0, step=0.25, value=7.5, label="Guidance Scale")
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="Euler")
                seed = gr.Slider(minimum=-1, maximum=9999999999, step=1, value=2313123, label="Seed", randomize=True)
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Illusion Diffusion Output")
            
    run_btn.click(
        inference,
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, seed, sampler],
        outputs=[result_image]
    )

app.queue(max_size=20)

if __name__ == "__main__":
    app.launch()