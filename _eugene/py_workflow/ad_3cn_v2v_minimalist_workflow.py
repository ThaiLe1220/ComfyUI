import os
import random
import sys
from typing import Sequence, Mapping, Any, Union, List, Tuple
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


def read_data_file(data_file_path: str) -> List[Tuple[str, str]]:
    video_prompts = []
    with open(data_file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("|")
            if len(parts) == 3:
                video_id, resolution, prompt = parts
                video_prompts.append((video_id, prompt))
    return video_prompts


from nodes import NODE_CLASS_MAPPINGS, VAEEncode


def generate_video_from_prompt(video_path: str, positive_prompt: str):

    lora_stacker = NODE_CLASS_MAPPINGS["LoRA Stacker"]()
    lora_stacker_40 = lora_stacker.lora_stacker(
        input_mode="advanced",
        lora_count=1,
        lora_name_1="add_detail.safetensors",
        model_str_1=0.5,
        clip_str_1=0,
    )

    controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
    controlnetloaderadvanced_43 = controlnetloaderadvanced.load_controlnet(
        control_net_name="control_v11p_sd15_lineart_fp16.safetensors"
    )

    controlnetloaderadvanced_49 = controlnetloaderadvanced.load_controlnet(
        control_net_name="control_v11p_sd15_softedge_fp16.safetensors"
    )

    controlnetloaderadvanced_52 = controlnetloaderadvanced.load_controlnet(
        control_net_name="control_v11f1p_sd15_depth_fp16.safetensors"
    )

    ade_loadanimatediffmodel = NODE_CLASS_MAPPINGS["ADE_LoadAnimateDiffModel"]()
    ade_loadanimatediffmodel_83 = ade_loadanimatediffmodel.load_motion_model(
        model_name="AnimateLCM_sd15_t2v.ckpt"
    )

    ade_animatediffsamplingsettings = NODE_CLASS_MAPPINGS[
        "ADE_AnimateDiffSamplingSettings"
    ]()
    ade_animatediffsamplingsettings_86 = (
        ade_animatediffsamplingsettings.create_settings(
            batch_offset=0,
            noise_type="default",
            seed_gen="comfy",
            seed_offset=0,
            adapt_denoise_steps=True,
        )
    )

    ade_loopeduniformcontextoptions = NODE_CLASS_MAPPINGS[
        "ADE_LoopedUniformContextOptions"
    ]()
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

    layerutility_purgevram = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM"]()
    ade_applyanimatediffmodelsimple = NODE_CLASS_MAPPINGS[
        "ADE_ApplyAnimateDiffModelSimple"
    ]()
    ade_useevolvedsampling = NODE_CLASS_MAPPINGS["ADE_UseEvolvedSampling"]()
    ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()
    vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

    ade_applyanimatediffmodelsimple_88 = (
        ade_applyanimatediffmodelsimple.apply_motion_model(
            motion_model=get_value_at_index(ade_loadanimatediffmodel_83, 0)
        )
    )

    with torch.inference_mode():
        vhs_loadvideopath = NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]()
        vhs_loadvideopath_10 = vhs_loadvideopath.load_video(
            video=video_path,
            force_rate=24,
            force_size="Disabled",
            custom_width=0,
            custom_height=0,
            frame_load_cap=120,
            skip_first_frames=24,
            select_every_nth=2,
        )

        jwimageresizebyshorterside = NODE_CLASS_MAPPINGS["JWImageResizeByShorterSide"]()
        jwimageresizebyshorterside_14 = jwimageresizebyshorterside.execute(
            size=512,
            interpolation_mode="nearest exact",
            image=get_value_at_index(vhs_loadvideopath_10, 0),
        )

        imagepass = NODE_CLASS_MAPPINGS["ImagePass"]()
        imagepass_15 = imagepass.passthrough(
            image=get_value_at_index(jwimageresizebyshorterside_14, 0)
        )

        get_resolution_crystools = NODE_CLASS_MAPPINGS["Get resolution [Crystools]"]()
        get_resolution_crystools_17 = get_resolution_crystools.execute(
            image=get_value_at_index(imagepass_15, 0), unique_id=13095250626247549117
        )

        depthanythingpreprocessor = NODE_CLASS_MAPPINGS["DepthAnythingPreprocessor"]()
        depthanythingpreprocessor_55 = depthanythingpreprocessor.execute(
            ckpt_name="depth_anything_vitl14.pth",
            resolution=512,
            image=get_value_at_index(imagepass_15, 0),
        )

        aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        aio_preprocessor_48 = aio_preprocessor.execute(
            preprocessor="HEDPreprocessor",
            resolution=512,
            image=get_value_at_index(imagepass_15, 0),
        )

        aio_preprocessor_37 = aio_preprocessor.execute(
            preprocessor="LineArtPreprocessor",
            resolution=512,
            image=get_value_at_index(imagepass_15, 0),
        )

        control_net_stacker = NODE_CLASS_MAPPINGS["Control Net Stacker"]()
        control_net_stacker_44 = control_net_stacker.control_net_stacker(
            strength=0.65,
            start_percent=0,
            end_percent=0.85,
            control_net=get_value_at_index(controlnetloaderadvanced_43, 0),
            image=get_value_at_index(aio_preprocessor_37, 0),
        )

        control_net_stacker_50 = control_net_stacker.control_net_stacker(
            strength=0.65,
            start_percent=0,
            end_percent=0.85,
            control_net=get_value_at_index(controlnetloaderadvanced_49, 0),
            image=get_value_at_index(aio_preprocessor_48, 0),
            cnet_stack=get_value_at_index(control_net_stacker_44, 0),
        )

        control_net_stacker_53 = control_net_stacker.control_net_stacker(
            strength=0.65,
            start_percent=0,
            end_percent=0.85,
            control_net=get_value_at_index(controlnetloaderadvanced_52, 0),
            image=get_value_at_index(depthanythingpreprocessor_55, 0),
            cnet_stack=get_value_at_index(control_net_stacker_50, 0),
        )

        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()
        efficient_loader_38 = efficient_loader.efficientloader(
            ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors",
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors",
            clip_skip=-1,
            lora_name="lcm/SD1.5/pytorch_lora_weights.safetensors",
            lora_model_strength=0.5,
            lora_clip_strength=0,
            positive=f"(realistic photo), {positive_prompt}",
            negative="(nsfw:1.25), (nipples:1.25), (low quality, worst quality:1.2), low-resolution, lowres, jpeg artifacts, compression artifacts, poorly drawn, downsampling, aliasing, distorted, pixelated, fake, hyper, glitch, distortion, text, watermark, signature, user name, artist name, moir pattern, blurry, glossy, ugly, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, duplicate, error, beginner, overexposed, high-contrast, bad-contrast, selfie, handy, phone, embedding:badhandv4, naked, nude, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4, deformed, distorted, disfigured:1.3, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, disgusting, amputation\n",
            token_normalization="mean",
            weight_interpretation="A1111",
            empty_latent_width=get_value_at_index(get_resolution_crystools_17, 0),
            empty_latent_height=get_value_at_index(get_resolution_crystools_17, 1),
            batch_size=1,
            lora_stack=get_value_at_index(lora_stacker_40, 0),
            cnet_stack=get_value_at_index(control_net_stacker_53, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_80 = vaeencode.encode(
            pixels=get_value_at_index(imagepass_15, 0),
            vae=get_value_at_index(efficient_loader_38, 4),
        )

        layerutility_purgevram_16 = layerutility_purgevram.purge_vram(
            purge_cache=True,
            purge_models=True,
            anything=get_value_at_index(imagepass_15, 0),
        )

        ade_useevolvedsampling_94 = ade_useevolvedsampling.use_evolved_sampling(
            beta_schedule="lcm >> sqrt_linear",
            model=get_value_at_index(efficient_loader_38, 0),
            m_models=get_value_at_index(ade_applyanimatediffmodelsimple_88, 0),
            context_options=get_value_at_index(ade_loopeduniformcontextoptions_89, 0),
            sample_settings=get_value_at_index(ade_animatediffsamplingsettings_86, 0),
        )

        ksampler_efficient_81 = ksampler_efficient.sample(
            seed=random.randint(1, 2**64),
            steps=6,
            cfg=1.5,
            sampler_name="lcm",
            scheduler="sgm_uniform",
            denoise=0.8,
            preview_method="none",
            vae_decode="true",
            model=get_value_at_index(ade_useevolvedsampling_94, 0),
            positive=get_value_at_index(efficient_loader_38, 1),
            negative=get_value_at_index(efficient_loader_38, 2),
            latent_image=get_value_at_index(vaeencode_80, 0),
            optional_vae=get_value_at_index(efficient_loader_38, 4),
        )

        vhs_videocombine_111 = vhs_videocombine.combine_video(
            frame_rate=24,
            loop_count=0,
            filename_prefix="mixkit_v2",
            format="video/h264-mp4",
            pingpong=False,
            save_output=True,
            images=get_value_at_index(ksampler_efficient_81, 5),
            unique_id=9989530071036380741,
        )

        layerutility_purgevram_138 = layerutility_purgevram.purge_vram(
            purge_cache=True,
            purge_models=True,
            anything=get_value_at_index(ksampler_efficient_81, 5),
        )


if __name__ == "__main__":
    data_file_path = (
        "/home/ubuntu/Desktop/Eugene/ComfyUI/input/__mixkit_v2__/data_c.txt"
    )
    output_dir = "/home/ubuntu/Desktop/Eugene/ComfyUI/output"
    video_prompts = read_data_file(data_file_path)

    import_custom_nodes()

    for index, (video_id, positive_prompt) in enumerate(video_prompts, start=1):
        output_file = os.path.join(output_dir, f"mixkit_v2_{index:05d}.mp4")

        if os.path.exists(output_file):
            print(f"[Workflow] Skipping already processed video: {video_id}")
            continue

        video_path = (
            f"/home/ubuntu/Desktop/Eugene/ComfyUI/input/__mixkit_v2__/{video_id}"
        )
        print(
            f"[Workflow] Processing video: {video_id}, Description: {positive_prompt}"
        )
        generate_video_from_prompt(video_path, positive_prompt)
        print(f"[Workflow] Finished processing video: {video_id}")

    print("All videos processed.")
