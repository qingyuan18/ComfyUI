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
import time

try:
    from comfy import model_management
except Exception:
    import sys
    sys.path.append("/home/ubuntu/ComfyUI")
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

svd_path =  "/home/ubuntu/ComfyUI/models/checkpoints/svd.safetensors"

# out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
xla_device = xm.xla_device()
out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True)

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


VAE_SCALING_FACTOR = 8



################## 3.2: unet compile  ##################
UNET_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "unet"
UNET_COMPILATION_DIR.mkdir(exist_ok=True)
    
##ouput unet model
unet_model=out[0].model.diffusion_model

unet_model = fd.make_forward_verbose(model=unet_model, model_name="U-Net")
unet = copy.deepcopy(unet_model)

del unet_model

UNET_IN_CHANNELS = unet.in_channels
# config["projection_dim"]
ENCODER_PROJECTION_DIM = clip_vision_model.model.visual_projection.out_features
MODEL_MAX_LENGTH = 512


example_input_sample = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, UNET_IN_CHANNELS, HEIGHT//VAE_SCALING_FACTOR, WIDTH//VAE_SCALING_FACTOR), dtype=DTYPE)
example_timestep = torch.randint(0, 1000, (BATCH_SIZE*NUM_IMAGES_PER_PROMPT,), dtype=DTYPE)
example_encoder_hidden_states = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, MODEL_MAX_LENGTH, ENCODER_PROJECTION_DIM), dtype=DTYPE)
example_y = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, ENCODER_PROJECTION_DIM), dtype=DTYPE)

example_inputs = (example_input_sample, example_timestep, example_encoder_hidden_states, example_y)

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
