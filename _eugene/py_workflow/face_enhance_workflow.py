import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

import re
import time


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_external_custom_nodes, init_builtin_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_external_custom_nodes()
    init_builtin_extra_nodes()


from nodes import VAEEncode, LoadImage, NODE_CLASS_MAPPINGS, SaveImage

import torch.cuda
import gc
import comfy.model_management
import time


def purge_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def purge_model() -> None:
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


import_custom_nodes()


def enhance_image(image_path, image_output_name):
    print(f"[Debug] Starting image generation for {image_path}")
    positive_prompt = "masterpiece,best quality,(photorealistic:1.1),8k raw photo,bokeh,detailed face,detailed skin,depth of field,"
    negative_prompt = "nsfw, naked, nude, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, drawing,paiting,crayon,sketch,graphite,impressionist,noisy,blurry,soft,deformed,ugly,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,(monochrome,greyscale,old photo),"

    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=image_path)

        # wd14taggerpysssss = NODE_CLASS_MAPPINGS["WD14Tagger|pysssss"]()
        # wd14taggerpysssss_27 = wd14taggerpysssss.tag(
        #     model="wd-vit-tagger-v3",
        #     threshold=0.3,
        #     character_threshold=0.85,
        #     replace_underscore="",
        #     exclude_tags="",
        #     image=get_value_at_index(loadimage_1, 0),
        # )
        # print("[Debug] WD14Tagger|pysssss successful")

        # print(wd14taggerpysssss_27["result"])

        tilepreprocessor = NODE_CLASS_MAPPINGS["TilePreprocessor"]()
        tilepreprocessor_281 = tilepreprocessor.execute(
            pyrUp_iters=1, resolution=1024, image=get_value_at_index(loadimage_1, 0)
        )

        lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        lineartpreprocessor_282 = lineartpreprocessor.execute(
            resolution=1024, image=get_value_at_index(loadimage_1, 0), coarse="disable"
        )

        cr_multi_controlnet_stack = NODE_CLASS_MAPPINGS["CR Multi-ControlNet Stack"]()
        cr_multi_controlnet_stack_20 = cr_multi_controlnet_stack.controlnet_stacker(
            switch_1="On",
            controlnet_1="control_v11f1e_sd15_tile_fp16.safetensors",
            controlnet_strength_1=1,
            start_percent_1=0,
            end_percent_1=1,
            switch_2="On",
            controlnet_2="control_v11p_sd15_lineart_fp16.safetensors",
            controlnet_strength_2=1,
            start_percent_2=0,
            end_percent_2=1,
            switch_3="Off",
            controlnet_3="None",
            controlnet_strength_3=1,
            start_percent_3=0,
            end_percent_3=1,
            image_1=get_value_at_index(tilepreprocessor_281, 0),
            image_2=get_value_at_index(lineartpreprocessor_282, 0),
        )

        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
        efficient_loader_14 = efficient_loader.efficientloader(
            ckpt_name="realisticVisionV60B1_v51VAE.safetensors",
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors",
            clip_skip=-1,
            lora_name="None",
            lora_model_strength=1,
            lora_clip_strength=1,
            # positive=f"{positive_prompt}{wd14taggerpysssss_27['result']}",
            positive=f"{positive_prompt}",
            negative=negative_prompt,
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=512,
            empty_latent_height=512,
            batch_size=1,
            cnet_stack=get_value_at_index(cr_multi_controlnet_stack_20, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_13 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_1, 0),
            vae=get_value_at_index(efficient_loader_14, 4),
        )

        ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()
        saveimage = SaveImage()

        ksampler_efficient_12 = ksampler_efficient.sample(
            seed=random.randint(1, 2**64),
            steps=6,
            cfg=1.5,
            sampler_name="dpmpp_sde",
            scheduler="karras",
            denoise=0.35,
            preview_method="auto",
            vae_decode="true",
            model=get_value_at_index(efficient_loader_14, 0),
            positive=get_value_at_index(efficient_loader_14, 1),
            negative=get_value_at_index(efficient_loader_14, 2),
            latent_image=get_value_at_index(vaeencode_13, 0),
            optional_vae=get_value_at_index(efficient_loader_14, 4),
        )

        purge_cache()
        purge_model()

        saveimage_308 = saveimage.save_images(
            filename_prefix=f"{image_output_name}",
            images=get_value_at_index(ksampler_efficient_12, 5),
        )


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)
    ]


def enhance_images_in_directory(directory_path, image_output_name, skip_count=0):
    # Get the list of files and sort them using the natural sort key
    filenames = sorted(os.listdir(directory_path), key=natural_sort_key)

    for index, filename in enumerate(filenames):
        if index < skip_count:
            continue
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            print(f"Enhancing image: {filename}")  # Print the image name
            image_path = os.path.join(directory_path, filename)

            start_time = time.time()  # Start timing
            enhance_image(image_path, image_output_name)
            end_time = time.time()  # End timing

            processing_time = end_time - start_time
            print(f"Processing time for {filename}: {processing_time:.2f} seconds")


# Assuming `enhance_image` is defined elsewhere in your code


if __name__ == "__main__":
    # print("Hello World!")
    # enhance_images_in_directory(
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/_v1.4/cropped_faces", "input"
    # )
    # enhance_images_in_directory(
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/_v1.4/restored_faces", "v1.4"
    # )

    # enhance_images_in_directory(
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/_v1024/restored_faces", "v1024", 23
    # )

    # enhance_images_in_directory(
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/_vRF/restored_faces", "vRF"
    # )

    enhance_images_in_directory("/home/ubuntu/Desktop/eugene/GFPGAN/_vTest", "vTest")
