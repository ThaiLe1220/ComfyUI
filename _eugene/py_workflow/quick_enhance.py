import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import time

import comfy.controlnet
import folder_paths

# Global variables
ctx = contextlib.nullcontext()
_custom_nodes_imported = False
_custom_path_added = False
NODE_CLASS_MAPPINGS = None

tilepreprocessor = None
lineartpreprocessor = None
cr_multi_controlnet_stack = None
efficient_loader = None
vaeencode = None
ksampler_efficient = None


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        return os.path.join(path, name)

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")
        print(f"'{comfyui_path}' added to sys.path")


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_builtin_extra_nodes()
    init_external_custom_nodes()


def add_extra_model_paths() -> None:
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def setup_environment():
    global _custom_path_added, _custom_nodes_imported, NODE_CLASS_MAPPINGS
    global tilepreprocessor, lineartpreprocessor, cr_multi_controlnet_stack
    global efficient_loader, vaeencode, ksampler_efficient

    if not _custom_path_added:
        add_comfyui_directory_to_sys_path()
        add_extra_model_paths()
        _custom_path_added = True
    if not _custom_nodes_imported:
        import_custom_nodes()
        _custom_nodes_imported = True

    from nodes import NODE_CLASS_MAPPINGS

    tilepreprocessor = NODE_CLASS_MAPPINGS["TilePreprocessor"]()
    lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
    cr_multi_controlnet_stack = NODE_CLASS_MAPPINGS["CR Multi-ControlNet Stack"]()
    efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
    vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()


PROMPT_DATA = json.loads(
    "{"
    '"12": {"inputs": {"seed": 497644448322528, "steps": 6, "cfg": 1.5, "sampler_name": "dpmpp_sde", "scheduler": "karras", "denoise": 0.3, "preview_method": "auto", "vae_decode": "true", "model": ["14", 0], "positive": ["14", 1], "negative": ["14", 2], "latent_image": ["13", 0], "optional_vae": ["14", 4]}, "class_type": "KSampler (Efficient)", "_meta": {"title": "KSampler (Efficient)"}}, '
    '"13": {"inputs": {"pixels": "image_tensor", "vae": ["14", 4]}, "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"}}, '
    '"14": {"inputs": {"ckpt_name": "realisticVisionV60B1_v51VAE.safetensors", "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors", "clip_skip": -1, "lora_name": "add_detail.safetensors", "lora_model_strength": 0.5, "lora_clip_strength": 0.5, "positive": ["88", 0], "negative": ["268", 0], "token_normalization": "none", "weight_interpretation": "comfy", "empty_latent_width": 512, "empty_latent_height": 512, "batch_size": 1, "cnet_stack": ["20", 0]}, "class_type": "Efficient Loader", "_meta": {"title": "Efficient Loader"}}, '
    '"20": {"inputs": {"switch_1": "On", "controlnet_1": "control_v11f1e_sd15_tile_fp16.safetensors", "controlnet_strength_1": 0.65, "start_percent_1": 0, "end_percent_1": 1.0, "switch_2": "On", "controlnet_2": "control_v11p_sd15_lineart_fp16.safetensors", "controlnet_strength_2": 0.65, "start_percent_2": 0, "end_percent_2": 1.0, "switch_3": "Off", "controlnet_3": "control_v11f1p_sd15_depth_fp16.safetensors", "controlnet_strength_3": 1.0, "start_percent_3": 0, "end_percent_3": 1.0, "image_1": ["281", 0], "image_2": ["282", 0]}, "class_type": "CR Multi-ControlNet Stack", "_meta": {"title": "\\ud83d\\udd79\\ufe0f CR Multi-ControlNet Stack"}}, '
    '"88": {"inputs": {"prompt": "masterpiece,best quality,(photorealistic:1.1),8k raw photo,bokeh,detailed face\\uff0cdetailed skin,depth of field,"}, "class_type": "CR Prompt Text", "_meta": {"title": "Quality"}}, '
    '"268": {"inputs": {"prompt": "(nsfw, naked, nude, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, drawing,paiting,crayon,sketch,graphite,impressionist,noisy,blurry,soft,deformed,ugly,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,(monochrome,greyscale,old photo),\\n\\n"}, "class_type": "CR Prompt Text", "_meta": {"title": "Negative"}}, '
    '"281": {"inputs": {"pyrUp_iters": 1, "resolution": 512, "image": "image_tensor"}, "class_type": "TilePreprocessor", "_meta": {"title": "Tile"}}, '
    '"282": {"inputs": {"coarse": "disable", "resolution": 512, "image": "image_tensor"}, "class_type": "LineArtPreprocessor", "_meta": {"title": "Realistic Lineart"}}'
    "}"
)


def load_image(image_path):
    img = Image.open(image_path)

    output_images = []
    w, h = None, None

    excluded_formats = ["MPO"]

    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)

        if i.mode == "I":
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w, h = image.size

        if image.size != (w, h):
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        output_images.append(image)

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
    else:
        output_image = output_images[0]

    print(output_image)

    return output_image


def create_controlnet_list(
    controlnet_1,
    controlnet_strength_1,
    image_1,
    start_percent_1,
    end_percent_1,
    controlnet_2,
    controlnet_strength_2,
    image_2,
    start_percent_2,
    end_percent_2,
):

    controlnet_list = []

    # Helper function to load and add ControlNet to the list
    def add_controlnet(name, strength, image, start_percent, end_percent):
        if name != "None" and image is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", name)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.append(
                (controlnet, image, strength, start_percent, end_percent)
            )

    # Add ControlNet 1
    add_controlnet(
        controlnet_1, controlnet_strength_1, image_1, start_percent_1, end_percent_1
    )

    # Add ControlNet 2
    add_controlnet(
        controlnet_2, controlnet_strength_2, image_2, start_percent_2, end_percent_2
    )

    return controlnet_list


def save_image(output_image, output_filename):
    # Convert the PyTorch tensor to a numpy array
    if isinstance(output_image, torch.Tensor):
        image_np = output_image.cpu().numpy()
    else:
        image_np = output_image

    # Ensure the image is in the correct format (H, W, C) and scale to 0-255
    if image_np.ndim == 4:
        image_np = image_np.squeeze(0)  # Remove batch dimension if present
    if image_np.shape[0] == 3:
        image_np = np.transpose(
            image_np, (1, 2, 0)
        )  # Change from (C, H, W) to (H, W, C)

    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # Create a PIL Image
    img = Image.fromarray(image_np)

    # Save the image
    img.save(output_filename, format="PNG")
    print(f"Saved image to {output_filename}")


def enhance_image(image_path, positive_prompt, negative_prompt):
    with torch.inference_mode():
        image_tensor = load_image(image_path)

        tilepreprocessor_281 = tilepreprocessor.execute(
            pyrUp_iters=1, resolution=512, image=image_tensor
        )

        lineartpreprocessor_282 = lineartpreprocessor.execute(
            coarse="disable", resolution=512, image=image_tensor
        )

        controlnet_list = create_controlnet_list(
            "control_v11f1e_sd15_tile_fp16.safetensors",
            0.65,
            get_value_at_index(tilepreprocessor_281, 0),
            0.0,
            1.0,
            "control_v11p_sd15_lineart_fp16.safetensors",
            0.65,
            get_value_at_index(lineartpreprocessor_282, 0),
            0.0,
            1.0,
        )

        efficient_loader_14 = efficient_loader.efficientloader(
            ckpt_name="realisticVisionV60B1_v51VAE.safetensors",
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors",
            clip_skip=-1,
            lora_name="add_detail.safetensors",
            lora_model_strength=0.5,
            lora_clip_strength=0.5,
            positive=positive_prompt,
            negative=negative_prompt,
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=512,
            empty_latent_height=512,
            batch_size=1,
            cnet_stack=controlnet_list,
            prompt=PROMPT_DATA,
        )

        vaeencode_13 = vaeencode.encode(
            pixels=image_tensor,
            vae=get_value_at_index(efficient_loader_14, 4),
        )

        ksampler_efficient_12 = ksampler_efficient.sample(
            seed=random.randint(1, 2**64),
            steps=6,
            cfg=1.5,
            sampler_name="dpmpp_sde",
            scheduler="karras",
            denoise=0.3,
            preview_method="auto",
            vae_decode="true",
            model=get_value_at_index(efficient_loader_14, 0),
            positive=get_value_at_index(efficient_loader_14, 1),
            negative=get_value_at_index(efficient_loader_14, 2),
            latent_image=get_value_at_index(vaeencode_13, 0),
            optional_vae=get_value_at_index(efficient_loader_14, 4),
            prompt=PROMPT_DATA,
        )

        return get_value_at_index(ksampler_efficient_12, 5)


def main():
    start_time = time.time()
    # setup_environment()
    setup_time = time.time() - start_time
    print(f"Time to setup environment: {setup_time:.2f} seconds")

    image_path = (
        "/home/ubuntu/Desktop/eugene/GFPGAN/_v1024/restored_faces/image9_00.png"
    )
    positive_prompt = "masterpiece, best quality, (photorealistic:1.1), 8k raw photo, bokeh, beautifully detailed face, clear eyes, smooth skin texture, natural beauty, depth of field, expressive features"
    negative_prompt = "(nsfw,naked,nude,deformed iris,deformed pupils,semi-realistic,cgi,3d,render,sketch,cartoon,anime),(deformed,distorted,disfigured:1.3),poorly drawn,mutated,ugly,disgusting,amputation,drawing,paiting,crayon,sketch,graphite,impressionist,noisy,blurry,soft,deformed,ugly,lowers,bad anatomy,text,error,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username"

    # Assuming load_image is a function that loads an image and returns a tensor
    image_tensor = load_image(image_path)

    # Check the characteristics of the tensor
    print(f"Shape: {image_tensor.shape}")
    print(f"Data Type: {image_tensor.dtype}")

    # Additional statistics
    print(f"Min: {torch.min(image_tensor).item()}")
    print(f"Max: {torch.max(image_tensor).item()}")
    print(f"Mean: {torch.mean(image_tensor).item()}")

    # for i in range(1):
    #     start_time = time.time()
    #     output_image = enhance_image(image_path, positive_prompt, negative_prompt)
    #     save_image(output_image, f"image_{i+1}.png")
    #     iteration_time = time.time() - start_time
    #     print(f"Iteration {i+1} time: {iteration_time:.2f} seconds")


if __name__ == "__main__":
    main()
