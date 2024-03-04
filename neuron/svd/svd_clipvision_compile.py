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

try: 
    svd_path =  "/home/ubuntu/ComfyUI/models/checkpoints/svd.safetensors"
    # out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
    out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True)
except Exception:
    svd_path = "/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/a1ce917313331d9d6cdea065aa176c27198bcaad/svd_xt.safetensors"
    out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True)

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


################# 3.3: clip vison compile ##################
CLIP_VISION_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_VISION"
CLIP_VISION_COMPILATION_DIR.mkdir(exist_ok=True)

clip_vision_vision_model = copy.deepcopy(clip_vision_model.model.vision_model)
clip_vision_visual_projection = copy.deepcopy(clip_vision_model.model.visual_projection)

# VISION_MODEL_HIDDEN_DIM = clip_vision_vision_model.embed_dim
# temp for debug 
VISION_MODEL_HIDDEN_DIM = 1280
VAE_OUT_CHANNELS = 3
HEIGHT = 224
WIDTH = 224

del clip_vision_model
example_clip_vision_vision_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
example_clip_vision_visual_projection_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VISION_MODEL_HIDDEN_DIM), dtype=DTYPE)

# ## test directly torch trace
# example_kwarg_inputs = {"pixel_values":example_clip_vision_vision_model_input}
# with torch.no_grad():
#    traced_model = torch.jit.trace(clip_vision_vision_model, example_kwarg_inputs=example_kwarg_inputs)
# print("torch jit compile success!")
# print(traced_model.graph)
# traced_model.save("/var/tmp/traced_clip_vision_vision_model.pt")

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

for neuron_model, file_name in zip((clip_vision_vision_model_neuron, clip_vision_visual_projection_neuron), ("clip_vision_vision_model.pt", "clip_vision_visual_projection.pt")):
   torch_neuronx.async_load(neuron_model)
   torch_neuronx.lazy_load(neuron_model)
   torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)
# Free up memory
del clip_vision_vision_model, example_clip_vision_vision_model_input
del clip_vision_visual_projection_neuron, example_clip_vision_visual_projection_input
# del clip_vision_vision_model, example_clip_vision_vision_model_input, clip_vision_visual_projection, example_clip_vision_visual_projection_input
