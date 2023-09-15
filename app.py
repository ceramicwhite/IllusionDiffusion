import torch
import os
import gradio as gr
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)

# Initialize both pipelines
init_pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16).to("cuda")
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)
main_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V2.0",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")

# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

# Inference function
def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    strength: float = 0.9,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")
    
    # Generate the initial image
    init_image = init_pipe(prompt).images[0]

    # Rest of your existing code
    control_image = control_image.resize((512, 512))
    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=control_image,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        strength=strength,
        num_inference_steps=30,
    )
    return out.images[0]

with gr.Blocks() as app:
    gr.Markdown(
        '''
        <center>

        <span style="color:blue; font-size:32px;"># Illusion Diffusion 🌀</span>  
        <span style="color:black; font-size:24px;">## Generate stunning illusion artwork with Stable Diffusion</span>  
        <span style="color:black; font-size:20px;">**A space by AP [Follow me on Twitter](https://twitter.com/angrypenguinPNG)**</span>  
        
        </center>

        <p style="text-align:center;">
        *This project works by using the QR Control Net by Monster Labs: [Monster Labs QR Control Net](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster).*
        *Given a prompt, we generate an init image and pass that alongside the control illusion to create a stunning illusion!* 
        *Credit to : [MrUgleh](https://twitter.com/MrUgleh) for discovering the workflow :)*
        </p>

        '''
    )
    
    with gr.Row():
        with gr.Column():
            control_image = gr.Image(label="Input Illusion", type="pil")
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
        inputs=[control_image, prompt, negative_prompt, guidance_scale, controlnet_conditioning_scale, strength, seed, sampler],
        outputs=[result_image]
    )

app.queue(concurrency_count=4, max_size=20)

if __name__ == "__main__":
    app.launch(debug=True)