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
    ControlNetApplyAdvanced,
    NODE_CLASS_MAPPINGS,
    VAELoader,
    KSamplerAdvanced,
    VAEDecode,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    ImageScale,
    VAEEncode,
)


def read_video_params(file_path: str):
    video_params = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) == 3:
                video_id, resolution, description = parts
                width, height = map(int, resolution.split("x"))
                video_params.append(
                    {
                        "video_id": video_id,
                        "description": description,
                        "width": width,
                        "height": height,
                    }
                )
    return video_params


def main():
    import_custom_nodes()

    # input_file = "/home/ubuntu/Desktop/Eugene/ComfyUI/input/eugene/data_c.txt"
    input_file = "/home/ubuntu/Desktop/Eugene/ComfyUI/input/__mixkit__/data_c.txt"
    video_params = read_video_params(input_file)

    for params in video_params:
        video_id = params["video_id"]
        description = params["description"]
        width = int(params["width"] * 3 / 4)
        height = int(params["height"] * 3 / 4)

        with torch.inference_mode():
            vaeloader = VAELoader()
            vaeloader_2 = vaeloader.load_vae(
                vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
            )

            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_110 = checkpointloadersimple.load_checkpoint(
                ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors"
            )

            cliptextencode = CLIPTextEncode()
            cliptextencode_3 = cliptextencode.encode(
                text=description,
                clip=get_value_at_index(checkpointloadersimple_110, 1),
            )

            cliptextencode_6 = cliptextencode.encode(
                text="(nsfw, naked, nude, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
                clip=get_value_at_index(checkpointloadersimple_110, 1),
            )

            vhs_loadvideopath = NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]()
            vhs_loadvideopath_125 = vhs_loadvideopath.load_video(
                video=f"/home/ubuntu/Desktop/Eugene/ComfyUI/input/__mixkit__/{video_id}",
                force_rate=24,
                force_size="Disabled",
                custom_width=width,
                custom_height=height,
                frame_load_cap=60,
                skip_first_frames=60,
                select_every_nth=1,
            )

            imagescale = ImageScale()
            imagescale_53 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=width,
                height=height,
                crop="disabled",
                image=get_value_at_index(vhs_loadvideopath_125, 0),
            )

            vaeencode = VAEEncode()
            vaeencode_56 = vaeencode.encode(
                pixels=get_value_at_index(imagescale_53, 0),
                vae=get_value_at_index(vaeloader_2, 0),
            )

            controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
            controlnetloaderadvanced_70 = controlnetloaderadvanced.load_controlnet(
                control_net_name="control_v11f1p_sd15_depth_fp16.safetensors"
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

            controlnetloaderadvanced_97 = controlnetloaderadvanced.load_controlnet(
                control_net_name="control_v11p_sd15_openpose_fp16.safetensors"
            )

            ade_animatediffloaderwithcontext = NODE_CLASS_MAPPINGS[
                "ADE_AnimateDiffLoaderWithContext"
            ]()
            midas_depthmappreprocessor = NODE_CLASS_MAPPINGS[
                "MiDaS-DepthMapPreprocessor"
            ]()
            controlnetapplyadvanced = ControlNetApplyAdvanced()
            dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
            ksampleradvanced = KSamplerAdvanced()
            vaedecode = VAEDecode()
            vhs_splitimages = NODE_CLASS_MAPPINGS["VHS_SplitImages"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

            ade_animatediffloaderwithcontext_93 = (
                ade_animatediffloaderwithcontext.load_mm_and_inject_params(
                    model_name="temporaldiff-v1-animatediff.ckpt",
                    beta_schedule="sqrt_linear (AnimateDiff)",
                    motion_scale=1,
                    apply_v2_models_properly=False,
                    model=get_value_at_index(checkpointloadersimple_110, 0),
                    context_options=get_value_at_index(
                        ade_animatediffuniformcontextoptions_94, 0
                    ),
                )
            )

            midas_depthmappreprocessor_102 = midas_depthmappreprocessor.execute(
                a=6.28,
                bg_threshold=0.1,
                resolution=512,
                image=get_value_at_index(imagescale_53, 0),
            )

            controlnetapplyadvanced_72 = controlnetapplyadvanced.apply_controlnet(
                strength=0.6,
                start_percent=0,
                end_percent=0.95,
                positive=get_value_at_index(cliptextencode_3, 0),
                negative=get_value_at_index(cliptextencode_6, 0),
                control_net=get_value_at_index(controlnetloaderadvanced_70, 0),
                image=get_value_at_index(midas_depthmappreprocessor_102, 0),
            )

            dwpreprocessor_100 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                image=get_value_at_index(imagescale_53, 0),
            )

            controlnetapplyadvanced_99 = controlnetapplyadvanced.apply_controlnet(
                strength=0.9,
                start_percent=0,
                end_percent=0.95,
                positive=get_value_at_index(controlnetapplyadvanced_72, 0),
                negative=get_value_at_index(controlnetapplyadvanced_72, 1),
                control_net=get_value_at_index(controlnetloaderadvanced_97, 0),
                image=get_value_at_index(dwpreprocessor_100, 0),
            )

            ksampleradvanced_111 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=4444444444,
                steps=6,
                cfg=1.5,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(ade_animatediffloaderwithcontext_93, 0),
                positive=get_value_at_index(controlnetapplyadvanced_99, 0),
                negative=get_value_at_index(controlnetapplyadvanced_99, 1),
                latent_image=get_value_at_index(vaeencode_56, 0),
            )

            vaedecode_10 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_111, 0),
                vae=get_value_at_index(vaeloader_2, 0),
            )

            vhs_videocombine_109 = vhs_videocombine.combine_video(
                frame_rate=12,
                loop_count=0,
                filename_prefix=f"Mixkit_AD_{video_id}",
                format="video/h264-mp4",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode_10, 0),
                unique_id=8303483221727228961,
            )


if __name__ == "__main__":
    main()
