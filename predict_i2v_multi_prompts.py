

import json
import os

import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image

from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# Config and model path
model_name          = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM_Cog" and "DDIM_Origin"
sampler_name        = "DDIM_Origin"

# Load pretrained model if need
transformer_path    = "output_dir/checkpoint-3000/transformer/diffusion_pytorch_model.safetensors" 
vae_path            = None

# Other params
sample_size         = [224, 288]
video_length        = 29
fps                 = 2

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "close_the_drawer.jpg"
validation_image_end    = None

# prompts
insts                   = [
    "push the switch upwards",
    "turn on the yellow lamp",
    "grasp the pink block and turn it right",
    "lift the red block lying on the shelf",
    "rotate the red block 90 degrees to the right",
    "push the desk far from the camera",
]
prompt                  = "The video conforms to the laws of physics. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
negative_prompt         = "The video does not fit the laws of physics. The video is not of a high quality, it has a low resolution. Watermark present in each frame. Strange motion trajectory. "
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
save_path               = "samples/cogvideox-fun-videos_i2v"

transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLCogVideoX.from_pretrained(
    model_name, 
    subfolder="vae"
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

text_encoder = T5EncoderModel.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)
# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
else:
    pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )
if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

for inst in insts:
    with torch.no_grad():
        sample = pipeline(
            inst + prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,

            video        = input_video,
            mask_video   = input_video_mask
        ).videos

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    prefix = inst

    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
