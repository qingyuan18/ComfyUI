import torch
import time
import numpy as np
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
#import neuron.forward_decorator as fd
import comfy.sd

svd_path =  "/home/ubuntu/ComfyUI/models/checkpoints/svd.safetensors"

# out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
out=comfy.sd.load_checkpoint_guess_config(svd_path, output_vae=True, output_clip=False, output_clipvision=True)


## 0 相关输入参数
HEIGHT = WIDTH = 512
DTYPE = torch.float16
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
    f"--auto-cast-type=fp16"
]
NEURON_COMPILER_CLI_ARGS = [
    "--target=inf2",
    "--enable-fast-loading-neuron-binaries",
    "--optlevel=1",
    *NEURON_COMPILER_TYPE_CASTING_CONFIG,
]
os.environ["NEURON_FUSE_SOFTMAX"] = "1"


VAE_SCALING_FACTOR = 8



################## 3.2: unet compile  ##################
UNET_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "unet"
UNET_COMPILATION_DIR.mkdir(exist_ok=True)
    
##ouput unet model
unet_model=out[0].model.diffusion_model

#unet_model = fd.make_forward_verbose(model=unet_model, model_name="U-Net")
unet = copy.deepcopy(unet_model)

del unet_model

UNET_IN_CHANNELS = unet.in_channels

import torch


# 位置参数
#x = torch.randn((14, UNET_IN_CHANNELS, HEIGHT, WIDTH),dtype=DTYPE,device=xm.xla_device())  # 假设的输入数据
x = torch.randn((4, 8, 72, 128))  # 假设的输入数据
timesteps = torch.randint(low=0, high=10, size=(4,))  # 假设的时间步或其他一维特征

# 关键字参数
context = torch.randn((4,1, 1024))
control = None
transformer_options = {
    'cond_or_uncond': [0],
    'sigmas': torch.tensor([3.5664] * 14)
}
y = torch.randn((4, 768))
image_only_indicator = torch.tensor([1],dtype=torch.int64,device=xm.xla_device())
num_video_frames = 14

# 构造sample_input
example_kwarg_inputs = {"x":x, "timesteps":timesteps,"context":context, "y":y
                 #"control":control,
                 #"transformer_options":transformer_options,
                 #'image_only_indicator':image_only_indicator,
                 #'num_video_frames':num_video_frames
                 }

example_inputs = (x, timesteps,context,y)
### test directly torch trace
#with torch.no_grad():
#    traced_model = torch.jit.trace(unet, example_kwarg_inputs=example_kwarg_inputs)
#print("torch jit compile success!")
#print(traced_model.graph)
#traced_model.save("/var/tmp/traced_unet.pt")


with torch.no_grad():
    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs=example_inputs,
        compiler_workdir=UNET_COMPILATION_DIR,
        compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={UNET_COMPILATION_DIR}/log-neuron-cc.txt', "--model-type=unet-inference"],
    )

# Free up memory
del x, timesteps,context,y, example_inputs, unet
print(unet_neuron.code)

# save compiled unet model
compiled_unet_filename = os.path.join(NEURON_COMPILER_OUTPUT_DIR, 'svd_unet_neuron.pt')
torch_neuronx.async_load(unet_neuron)
torch.jit.save(unet_neuron, compiled_unet_filename)
