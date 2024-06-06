import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
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
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    VAELoader,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    KSamplerAdvanced,
    VAEDecode,
)


def read_prompts(file_path: str) -> list:
    """Reads prompts from a given file, each line is considered as a separate prompt."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def main():
    # Read prompts from files
    positive_prompts = read_prompts(
        "/home/ubuntu/Desktop/Eugene/ComfyUI/input/eu/positive.txt"
    )
    negative_prompts = read_prompts(
        "/home/ubuntu/Desktop/Eugene/ComfyUI/input/eu/negative.txt"
    )

    import_custom_nodes()
    with torch.inference_mode():
        vaeloader = VAELoader()
        vaeloader_2 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_102 = checkpointloadersimple.load_checkpoint(
            ckpt_name="dreamshaper_8.safetensors"
        )

        ade_animatediffuniformcontextoptions = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffUniformContextOptions"
        ]()
        ade_animatediffuniformcontextoptions_94 = (
            ade_animatediffuniformcontextoptions.create_options(
                context_schedule="uniform",
                fuse_method="flat",
                context_length=16,
                context_stride=1,
                context_overlap=4,
                closed_loop=False,
            )
        )

        ade_emptylatentimagelarge = NODE_CLASS_MAPPINGS["ADE_EmptyLatentImageLarge"]()
        ade_emptylatentimagelarge_100 = ade_emptylatentimagelarge.generate(
            width=512, height=512, batch_size=60
        )

        ade_animatediffloaderwithcontext = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffLoaderWithContext"
        ]()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for positive_prompt, negative_prompt in zip(positive_prompts, negative_prompts):
            cliptextencode = CLIPTextEncode()
            cliptextencode_3 = cliptextencode.encode(
                text=positive_prompt,
                clip=get_value_at_index(checkpointloadersimple_102, 1),
            )

            cliptextencode_6 = cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(checkpointloadersimple_102, 1),
            )

            ade_animatediffloaderwithcontext_93 = (
                ade_animatediffloaderwithcontext.load_mm_and_inject_params(
                    model_name="mm_sd_v15_v2.ckpt",
                    beta_schedule="sqrt_linear (AnimateDiff)",
                    motion_scale=1,
                    apply_v2_models_properly=False,
                    model=get_value_at_index(checkpointloadersimple_102, 0),
                    context_options=get_value_at_index(
                        ade_animatediffuniformcontextoptions_94, 0
                    ),
                )
            )

            ksampleradvanced_107 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=444488884444,
                steps=32,
                cfg=8,
                sampler_name="dpmpp_2m_sde_gpu",
                scheduler="karras",
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(ade_animatediffloaderwithcontext_93, 0),
                positive=get_value_at_index(cliptextencode_3, 0),
                negative=get_value_at_index(cliptextencode_6, 0),
                latent_image=get_value_at_index(ade_emptylatentimagelarge_100, 0),
            )

            vaedecode_10 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_107, 0),
                vae=get_value_at_index(vaeloader_2, 0),
            )

            vhs_videocombine_101 = vhs_videocombine.combine_video(
                frame_rate=12,
                loop_count=0,
                filename_prefix="Eu",
                format="video/h264-mp4",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode_10, 0),
                unique_id=random.randint(1, 2**64),
            )


if __name__ == "__main__":
    main()
