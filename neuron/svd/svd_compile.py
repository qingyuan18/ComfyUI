import torch
import time
import numpy as np
import torch
import torch.jit
import torch.nn
import torch_neuronx
import torch_xla
import torch_xla.core.xla_model as xm
from pathlib import Path
from transformers.modeling_outputs import BaseModelOutputWithPooling
import os
import copy

import contextlib
import math

try:
    from comfy import model_management
except Exception:
    import sys
    sys.path.append("/home/ubuntu/pytorch_inf2_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aws-xc-comfyui/reference/qingyuan18/ComfyUI")
    from comfy import model_management

# from .ldm.util import instantiate_from_config
import comfy.utils
# from ... import clip_vision
# from ... import gligen
# from ... import model_base
# from ... import model_detection

import comfy.model_patcher
import comfy.t2i_adapter.adapter
import comfy.supported_models_base
import neuron.forward_decorator as fd
import comfy.sd

svd_path = "/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/a1ce917313331d9d6cdea065aa176c27198bcaad/svd_xt.safetensors" 

# out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True)
##ouput unet model
unet_model=out[0].model.diffusion_model
unet = fd.make_forward_verbose(model=unet_model, model_name="U-Net")

##output clip_vision model
clip_vision_model=out[3]
clip_vision_model.model = fd.make_forward_verbose(model=clip_vision_model.model , model_name="clip vision's vision_model)")
# clip_vision_model.visual_projection = fd.make_forward_verbose(model=pipe.safety_checker.visual_projection, model_name="clip vision visual_projection")

## output vae model
vae_model=out[2].first_stage_model
vae_model.decoder = fd.make_forward_verbose(model=vae_model.decoder, model_name="VAE (decoder)")
vae_model.encoder = fd.make_forward_verbose(model=vae_model.encoder, model_name="VAE (encoder)")

## 0 相关输入参数
HEIGHT = WIDTH = 512
DTYPE = torch.float32
BATCH_SIZE = 1
NUM_IMAGES_PER_PROMPT = 1

## 1:模型编译路径
NEURON_COMPILER_WORKDIR = Path("neuron_compiler_workdir")
NEURON_COMPILER_WORKDIR.mkdir(exist_ok=True)
NEURON_COMPILER_OUTPUT_DIR = Path("compiled_models")
NEURON_COMPILER_OUTPUT_DIR.mkdir(exist_ok=True)


## 2: 模型编译参数
NEURON_COMPILER_TYPE_CASTING_CONFIG = [
    "--auto-cast=matmult",
    f"--auto-cast-type=bf16"
]
NEURON_COMPILER_CLI_ARGS = [
    "--target=inf2",
    "--enable-fast-loading-neuron-binaries",
    *NEURON_COMPILER_TYPE_CASTING_CONFIG,
]
os.environ["NEURON_FUSE_SOFTMAX"] = "1"



##################  3.1: vae compile ##################
VAE_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "vae"
VAE_COMPILATION_DIR.mkdir(exist_ok=True)

vae_encoder = copy.deepcopy(vae_model.encoder)
vae_encoder = vae_encoder.to(xm.xla_device())
vae_decoder = copy.deepcopy(vae_model.decoder)

LATENT_CHANNELS = vae_encoder.in_channels
# VAE_SCALING_FACTOR = 2**(len(vae_model.encoder.out_ch)-1)
# VAE_SCALING_FACTOR = 8

del vae_model

# example_latent_sample = torch.randn((1, LATENT_CHANNELS, HEIGHT//VAE_SCALING_FACTOR, WIDTH//VAE_SCALING_FACTOR), dtype=DTYPE)

vae_encoder_example_input = torch.randn((1, LATENT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE, device=xm.xla_device())

batch_number = 7
VAE_SCALING_FACTOR = 8
vae_decoder_example_input = torch.randn((1, 4, HEIGHT//VAE_SCALING_FACTOR, WIDTH//VAE_SCALING_FACTOR), dtype=DTYPE, device=xm.xla_device())

with torch.no_grad():
    VAE_ENCODER_COMPILATION_DIR = VAE_COMPILATION_DIR / "encoder"
    vae_encoder_neuron = torch_neuronx.trace(
        vae_encoder,
        vae_encoder_example_input,
        compiler_workdir=VAE_ENCODER_COMPILATION_DIR,
        compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VAE_ENCODER_COMPILATION_DIR}/log-neuron-cc.txt'],
    )

    VAE_DECODER_COMPILATION_DIR = VAE_COMPILATION_DIR / "decoder"
    vae_decoder_neuron = torch_neuronx.trace(
        vae_decoder,
        vae_decoder_example_input,
        compiler_workdir=VAE_DECODER_COMPILATION_DIR / "decoder",
        compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VAE_DECODER_COMPILATION_DIR}/log-neuron-cc.txt'],
    )
# Free up memory
del vae_encoder, vae_decoder, vae_decoder_example_input, vae_encoder_example_input
print(vae_decoder_neuron.code)
for neuron_model, file_name in zip((vae_encoder_neuron, vae_decoder_neuron), ("vae_encoder.pt", "vae_decoder.pt")):
    torch_neuronx.async_load(neuron_model)
    torch_neuronx.lazy_load(neuron_model)
    torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)


################## 3.2: unet compile  ##################
UNET_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "unet"
UNET_COMPILATION_DIR.mkdir(exist_ok=True)

def ensure_unet_forward_neuron_compilable(unet_model: UNetModel) -> UNetModel:
    def decorate_forward_method(f: Callable) -> Callable:
        def decorated_forward_method(*args, **kwargs) -> torch.Tensor:
            kwargs.update({"return_dict": False})
            output_sample, = f(*args, **kwargs)
            return output_sample
        return decorated_forward_method
    model.forward = decorate_forward_method(model.forward)
    return model

unet = copy.deepcopy(unet_model)
unet = ensure_unet_forward_neuron_compilable(unet)

UNET_IN_CHANNELS = unet.in_channels
ENCODER_PROJECTION_DIM = clip_vision_model.config["projection_dim"]
MODEL_MAX_LENGTH = 512


example_input_sample = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, UNET_IN_CHANNELS, HEIGHT//VAE_SCALING_FACTOR, WIDTH//VAE_SCALING_FACTOR), dtype=DTYPE)
example_timestep = torch.randint(0, 1000, (BATCH_SIZE*NUM_IMAGES_PER_PROMPT,), dtype=DTYPE)
example_encoder_hidden_states = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, MODEL_MAX_LENGTH, ENCODER_PROJECTION_DIM), dtype=DTYPE)
example_inputs = (example_input_sample, example_timestep, example_encoder_hidden_states)

with torch.no_grad():
    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        compiler_workdir=UNET_COMPILATION_DIR,
        compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={UNET_COMPILATION_DIR}/log-neuron-cc.txt', "--model-type=unet-inference"],
    )

# Free up memory
del example_input_sample, example_timestep, example_encoder_hidden_states, example_inputs, unet
print(unet_neuron.code)

################## 3.3: clip vison compile ##################
CLIP_VISION_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_VISION"
CLIP_VISION_COMPILATION_DIR.mkdir(exist_ok=True)

def ensure_vision_model_forward_neuron_compilable(model: CLIPVision) -> CLIPVision:
    def decorate_forward_method(f: Callable) -> Callable:
        def decorated_forward_method(*args, **kwargs) -> Tuple[torch.Tensor]:
            kwargs.update({"return_dict": False})
            output = f(*args, **kwargs)
            return output
        return decorated_forward_method
    model.forward = decorate_forward_method(model.forward)
    return model
clip_vision_vision_model = copy.deepcopy(clip_vision_model.CLIPVision)
clip_vision_visual_projection = copy.deepcopy(clip_vision_model.visual_projection)
clip_vision_vision_model = ensure_vision_model_forward_neuron_compilable(clip_vision_vision_model)

VISION_MODEL_HIDDEN_DIM = clip_vision_vision_model.embed_dim

del clip_vision_model
example_clip_vision_vision_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
example_clip_vision_visual_projection_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VISION_MODEL_HIDDEN_DIM), dtype=DTYPE)

with torch.no_grad(): 
    CLIP_VISION_VISION_MODEL_COMPILATION_DIR = CLIP_VISION_COMPILATION_DIR / "vision_model"
    clip_vision_vision_model_neuron = torch_neuronx.trace(
        clip_vision_vision_model,
        example_clip_vision_vision_model_input, 
            compiler_workdir=CLIP_VISION_VISION_MODEL_COMPILATION_DIR,
            compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={CLIP_VISION_VISION_MODEL_COMPILATION_DIR}/log-neuron-cc.txt'],
            )

    CLIP_VISION_VISUAL_PROJECTION_DIR = CLIP_VISION_COMPILATION_DIR / "visual_projection"
    clip_vision_visual_projection_neuron = torch_neuronx.trace(
        clip_vision_visual_projection,
        example_clip_vision_visual_projection_input, 
            compiler_workdir=CLIP_VISION_VISUAL_PROJECTION_DIR,
            compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={CLIP_VISION_VISUAL_PROJECTION_DIR}/log-neuron-cc.txt'],
            )

# Free up memory
del clip_vision_vision_model, example_clip_vision_vision_model_input, clip_vision_visual_projection, example_clip_vision_visual_projection_input
