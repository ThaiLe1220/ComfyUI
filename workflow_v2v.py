import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
import time
import argparse


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


from nodes import NODE_CLASS_MAPPINGS, VAEEncode


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


from typing import Sequence, Mapping, Any, Union, List, Tuple


def read_data_file(data_file_path: str) -> List[Tuple[str, str]]:

    video_prompts = []
    with open(data_file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("|")
            if len(parts) == 3:
                video_name, prompt, human_presence = parts
                video_prompts.append((video_name, prompt, human_presence))
    return video_prompts


start_time = time.time()
import_custom_nodes()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[Workflow] Import Custome Node in {elapsed_time:.2f} seconds")

start_time = time.time()
vhs_loadvideopath = NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]()
get_resolution_crystools = NODE_CLASS_MAPPINGS["Get resolution [Crystools]"]()
aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
control_net_stacker = NODE_CLASS_MAPPINGS["Control Net Stacker"]()
ade_loadanimatediffmodel = NODE_CLASS_MAPPINGS["ADE_LoadAnimateDiffModel"]()
ade_animatediffsamplingsettings = NODE_CLASS_MAPPINGS[
    "ADE_AnimateDiffSamplingSettings"
]()
ade_useevolvedsampling = NODE_CLASS_MAPPINGS["ADE_UseEvolvedSampling"]()
ade_applyanimatediffmodelsimple = NODE_CLASS_MAPPINGS[
    "ADE_ApplyAnimateDiffModelSimple"
]()
ade_loopeduniformcontextoptions = NODE_CLASS_MAPPINGS[
    "ADE_LoopedUniformContextOptions"
]()
lora_stacker = NODE_CLASS_MAPPINGS["LoRA Stacker"]()
efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()
vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

# Setup Loras
lora_stacker_human = lora_stacker.lora_stacker(
    input_mode="advanced",
    lora_count=2,
    lora_name_1="add_detail.safetensors",
    model_str_1=0.3,
    clip_str_1=0.3,
    lora_name_2="depth_of_field_slider_v1.safetensors",
    model_str_2=1.2,
    clip_str_2=1.2,
)

lora_stacker_nonhuman = lora_stacker.lora_stacker(
    input_mode="advanced",
    lora_count=2,
    lora_name_1="add_detail.safetensors",
    model_str_1=0.5,
    clip_str_1=0.5,
    lora_name_2="depth_of_field_slider_v1.safetensors",
    model_str_2=1.5,
    clip_str_2=1.5,
)
# Load controlnet models
controlnetloaderadvanced_43 = controlnetloaderadvanced.load_controlnet(
    control_net_name="control_v11p_sd15_lineart_fp16.safetensors"
)
controlnetloaderadvanced_49 = controlnetloaderadvanced.load_controlnet(
    control_net_name="control_v11p_sd15_softedge_fp16.safetensors"
)
controlnetloaderadvanced_52 = controlnetloaderadvanced.load_controlnet(
    control_net_name="control_v11f1p_sd15_depth_fp16.safetensors"
)

# Setup Animatediff
ade_loadanimatediffmodel_83 = ade_loadanimatediffmodel.load_motion_model(
    model_name="AnimateLCM_sd15_t2v.ckpt"
)
ade_animatediffsamplingsettings_86 = ade_animatediffsamplingsettings.create_settings(
    batch_offset=0,
    noise_type="default",
    seed_gen="comfy",
    seed_offset=0,
    adapt_denoise_steps=True,
)
ade_loopeduniformcontextoptions_89 = ade_loopeduniformcontextoptions.create_options(
    context_length=16,
    context_stride=1,
    context_overlap=4,
    closed_loop=False,
    fuse_method="pyramid",
    use_on_equal_length=False,
    start_percent=0,
    guarantee_steps=1,
)
ade_applyanimatediffmodelsimple_88 = ade_applyanimatediffmodelsimple.apply_motion_model(
    motion_model=get_value_at_index(ade_loadanimatediffmodel_83, 0)
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"[Workflow] Setup Required Nodes in {elapsed_time:.2f} seconds")


def load_tensor(file_path):
    with open(file_path, "r") as f:
        # Read the first line to get the shape
        shape = tuple(map(int, f.readline().strip().split()))
        numpy_data = np.loadtxt(f)

    tensor = torch.tensor(numpy_data.reshape(shape))
    return {"samples": tensor}  # Return a dictionary with "samples" key


def load_tensor_np(file_path):
    numpy_data = np.load(file_path)
    tensor = torch.from_numpy(numpy_data)
    return {"samples": tensor}


def generate_video_from_prompt(
    video_path: str,
    latent_images_path: str,
    positive_prompt: str,
    control_net_params: dict,
    output_prefix: str,
    lora_stacker,
    debug: bool,
) -> None:
    if debug:
        print(f"[Debug] Start generation for {video_path}, prompt: {positive_prompt}")
        print(f"[Debug] Latent images path: {latent_images_path}")
        print(f"[Debug] Output prefix: {output_prefix}")
        print(f"[Debug] Control Net params: {control_net_params}")

    with torch.inference_mode():
        latent_images = load_tensor_np(latent_images_path)

        purge_cache()
        purge_model()

        vhs_loadvideopath_10 = vhs_loadvideopath.load_video(
            video=video_path,
            force_rate=24,
            force_size="Disabled",
            custom_width=0,
            custom_height=0,
            frame_load_cap=100,
            skip_first_frames=0,
            select_every_nth=1,
        )

        purge_cache()
        purge_model()

        get_resolution_crystools_17 = get_resolution_crystools.execute(
            image=get_value_at_index(vhs_loadvideopath_10, 0),
            unique_id=13095250626247549117,
        )

        control_net_params = control_net_params or {}

        def apply_control_net_stacker(
            control_net, image, cnet_stack=None, control_net_params=None
        ):
            control_net_params = control_net_params or {}
            strength = control_net_params.get("strength", 0.4)

            end_percent = control_net_params.get("end_percent", 0.7)
            return control_net_stacker.control_net_stacker(
                strength=strength,
                start_percent=0,
                end_percent=end_percent,
                control_net=control_net,
                image=image,
                cnet_stack=cnet_stack,
            )

        # Lineart Controlnet
        aio_preprocessor_37 = aio_preprocessor.execute(
            preprocessor="LineArtPreprocessor",
            resolution=512,
            image=get_value_at_index(vhs_loadvideopath_10, 0),
        )
        control_net_stacker_44 = apply_control_net_stacker(
            control_net=get_value_at_index(controlnetloaderadvanced_43, 0),
            image=get_value_at_index(aio_preprocessor_37, 0),
            control_net_params=control_net_params.get("lineart", {}),
        )

        # Softedge Controlnet
        aio_preprocessor_48 = aio_preprocessor.execute(
            preprocessor="HEDPreprocessor",
            resolution=512,
            image=get_value_at_index(vhs_loadvideopath_10, 0),
        )
        control_net_stacker_50 = apply_control_net_stacker(
            control_net=get_value_at_index(controlnetloaderadvanced_49, 0),
            image=get_value_at_index(aio_preprocessor_48, 0),
            cnet_stack=get_value_at_index(control_net_stacker_44, 0),
            control_net_params=control_net_params.get("softedge", {}),
        )

        # Depth Controlnet
        depthanythingpreprocessor_55 = depthanythingpreprocessor.execute(
            ckpt_name="depth_anything_vitl14.pth",
            resolution=512,
            image=get_value_at_index(vhs_loadvideopath_10, 0),
        )
        control_net_stacker_53 = apply_control_net_stacker(
            control_net=get_value_at_index(controlnetloaderadvanced_52, 0),
            image=get_value_at_index(depthanythingpreprocessor_55, 0),
            cnet_stack=get_value_at_index(control_net_stacker_50, 0),
            control_net_params=control_net_params.get("depth", {}),
        )

        efficient_loader_38 = efficient_loader.efficientloader(
            ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors",
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors",
            clip_skip=-1,
            lora_name="lcm/SD1.5/pytorch_lora_weights.safetensors",
            lora_model_strength=0.5,
            lora_clip_strength=0,
            positive=f"(realistic photo, 8k uhd), {positive_prompt}",
            negative="(nsfw:1.1), (nipples:1.1), (worst quality), (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream, orange",
            token_normalization="mean",
            weight_interpretation="A1111",
            empty_latent_width=get_value_at_index(get_resolution_crystools_17, 0),
            empty_latent_height=get_value_at_index(get_resolution_crystools_17, 1),
            batch_size=1,
            lora_stack=get_value_at_index(lora_stacker, 0),
            cnet_stack=get_value_at_index(control_net_stacker_53, 0),
        )

        purge_cache()
        purge_model()

        ade_useevolvedsampling_94 = ade_useevolvedsampling.use_evolved_sampling(
            beta_schedule="lcm >> sqrt_linear",
            model=get_value_at_index(efficient_loader_38, 0),
            m_models=get_value_at_index(ade_applyanimatediffmodelsimple_88, 0),
            context_options=get_value_at_index(ade_loopeduniformcontextoptions_89, 0),
            sample_settings=get_value_at_index(ade_animatediffsamplingsettings_86, 0),
        )

        ksampler_efficient_81 = ksampler_efficient.sample(
            seed=random.randint(1, 2**64),
            steps=5,
            cfg=1.5,
            sampler_name="lcm",
            scheduler="sgm_uniform",
            denoise=0.8,
            preview_method="none",
            vae_decode="true",
            model=get_value_at_index(ade_useevolvedsampling_94, 0),
            positive=get_value_at_index(efficient_loader_38, 1),
            negative=get_value_at_index(efficient_loader_38, 2),
            latent_image=latent_images,
            optional_vae=get_value_at_index(efficient_loader_38, 4),
        )

        purge_cache()
        purge_model()

        vhs_videocombine_111 = vhs_videocombine.combine_video(
            frame_rate=20,
            loop_count=0,
            filename_prefix=output_prefix,
            format="video/h264-mp4",
            pingpong=False,
            save_output=True,
            images=get_value_at_index(ksampler_efficient_81, 5),
            unique_id=9989530071036380741,
        )


if __name__ == "__main__":
    batch_number = 11
    data_file_path = f"input/bs1000_b{batch_number}/metadata_final_b{batch_number}.txt"
    video_base_path = f"input/bs1000_b{batch_number}/videos"
    latent_base_path = f"input/bs1000_b{batch_number}/latent_images"

    output_dir = f"output/v2v_videos_b{batch_number}"

    video_prompts = read_data_file(data_file_path)
    limited_prompts = video_prompts

    for index, (video_name, positive_prompt, human_presence) in enumerate(
        limited_prompts
    ):
        video_basename, ext = video_name.split(".")
        video_prefix, video_id = video_basename.split("_")
        output_file = os.path.join(
            output_dir, f"{video_prefix}_{int(video_id):06d}_00001.mp4"
        )

        if os.path.exists(output_file):
            print(
                f"[Workflow] Skipping already processed video (Index: {index}): {video_name}"
            )
            continue

        video_path = f"{video_base_path}/{video_name}"
        latent_images_path = f"{latent_base_path}/latent_{video_id}.npy"
        video_output_prefix = f"v2v_videos_b{batch_number}/processed_{video_id}"

        print(
            f"[Workflow] Processing video (Index: {index}): {video_name}, Description: {positive_prompt}, Human Presence: {human_presence}"
        )

        purge_cache()
        purge_model()

        start_time = time.time()
        if human_presence:
            control_net_params = {
                "lineart": {"strength": 0.55, "end_percent": 0.75},
                "softedge": {"strength": 0.6, "end_percent": 0.8},
                "depth": {"strength": 0.65, "end_percent": 0.85},
            }
            generate_video_from_prompt(
                video_path,
                latent_images_path,
                positive_prompt,
                control_net_params,
                video_output_prefix,
                lora_stacker_human,
                False,
            )
        else:
            control_net_params = {
                "lineart": {"strength": 0.45, "end_percent": 0.65},
                "softedge": {"strength": 0.5, "end_percent": 0.7},
                "depth": {"strength": 0.55, "end_percent": 0.75},
            }
            generate_video_from_prompt(
                video_path,
                latent_images_path,
                positive_prompt,
                control_net_params,
                video_output_prefix,
                lora_stacker_nonhuman,
                False,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"[Workflow] Finished processing video (Index: {index}): {video_name} in {elapsed_time:.2f} seconds\n"
        )

    print("All videos processed.")
