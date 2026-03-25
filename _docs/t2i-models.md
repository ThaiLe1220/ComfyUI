# Text-to-Image Models

4 models set up for T2I comparison on RTX 3090 (24 GB VRAM). All use 4 steps, 1024x1024.

## Quick Comparison

| Model | Params | Precision | VRAM | Steps | License | Workflow |
|-------|--------|-----------|------|-------|---------|----------|
| FLUX.1 [schnell] | 12B | FP8 | ~13 GB | 4 | Apache 2.0 | `eugene_flux1_schnell_fp8_t2i.json` |
| FLUX.2 [klein] 4B | 4B | BF16 | ~16 GB | 4 | Apache 2.0 | `eugene_flux2_klein4b_bf16_t2i.json` |
| FLUX.2 [klein] 9B | 9B | FP8 | ~18 GB | 4 | Non-commercial | `eugene_flux2_klein9b_fp8_t2i.json` (not yet downloaded) |
| Z-Image Turbo | 6B | FP8 | ~14.5 GB | 4 | Apache 2.0 | `eugene_zimage_turbo_fp8_t2i.json` |

## FLUX.1 [schnell]

By Black Forest Labs. 12B params, distilled for 4-step generation. Apache 2.0.

| File | Size | Path |
|------|------|------|
| `flux1-schnell-fp8.safetensors` | 17 GB | `checkpoints/` |

All-in-one checkpoint (bundles diffusion model + CLIP-L + T5-XXL + VAE). Simplest setup.

```
CheckpointLoaderSimple → MODEL, CLIP, VAE
CLIPTextEncode (positive) ← CLIP
CLIPTextEncode (empty) → ConditioningZeroOut
EmptyLatentImage (1024x1024)
KSampler (4 steps, CFG 1.0, euler, simple)
VAEDecode → SaveImage
```

Settings: steps=4, cfg=1.0, sampler=euler, scheduler=simple. Negative prompts ignored.

## FLUX.2 [klein] 4B

By Black Forest Labs. 4B params, step-distilled. Apache 2.0 — commercial use OK.

| File | Size | Path |
|------|------|------|
| `flux-2-klein-4b.safetensors` | 7.2 GB | `diffusion_models/` |
| `qwen_3_4b.safetensors` | 7.5 GB | `text_encoders/` |
| `flux2-vae.safetensors` | 321 MB | `vae/` |

```
UNETLoader + CLIPLoader (type=flux2) + VAELoader
CLIPTextEncode (positive + negative)
CFGGuider (cfg=1.0)
RandomNoise → KSamplerSelect (euler) → Flux2Scheduler (4 steps)
EmptyFlux2LatentImage (128ch, 1/16 downscale)
SamplerCustomAdvanced → VAEDecode → SaveImage
```

Settings: steps=4, cfg=1.0, sampler=euler, Flux2Scheduler (resolution-aware sigmas).

Note: There is also a "base" (non-distilled) variant for image editing that uses 20 steps and CFG 5.0. Our workflow uses the distilled version.

## FLUX.2 [klein] 9B

By Black Forest Labs. 9B params, step-distilled. **Non-commercial license** — testing only.

| File | Size | Path |
|------|------|------|
| `flux-2-klein-9b-fp8.safetensors` | 8.8 GB | `diffusion_models/` |
| `qwen_3_8b_fp8mixed.safetensors` | 8.1 GB | `text_encoders/` |
| `flux2-vae.safetensors` | 321 MB | `vae/` |

Same workflow structure as Klein 4B. Key differences:
- Uses Qwen3-8B text encoder (not 4B) — better language understanding
- FP8 quantized to fit 24 GB VRAM (BF16 would need ~29 GB)
- Requires HuggingFace authentication to download (gated repo)
- **Not yet downloaded.** Workflow is ready — run `huggingface-cli login` then download the two files to test

## Z-Image Turbo

By Tongyi-MAI (Alibaba). 6B params, step-distilled. Apache 2.0.

| File | Size | Path |
|------|------|------|
| `z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors` | 5.7 GB | `diffusion_models/` |
| `qwen_3_4b.safetensors` | 7.5 GB | `text_encoders/` |
| `ae.safetensors` | 320 MB | `vae/` |

```
UNETLoader + CLIPLoader (type=lumina2) + VAELoader
CLIPTextEncode (positive) → ConditioningZeroOut (negative)
EmptySD3LatentImage (16ch, 1/8 downscale)
ModelSamplingAuraFlow (shift=3.0)
KSampler (4 steps, CFG 1.0, res_multistep, simple)
VAEDecode → SaveImage
```

Settings: steps=4, cfg=1.0, sampler=res_multistep, scheduler=simple, AuraFlow shift=3.0.

**Critical:** CLIPLoader type must be `lumina2` (not `flux2`). Same `qwen_3_4b.safetensors` file, but different tokenizer.

## Architecture Differences

| | FLUX.1 Schnell | FLUX.2 Klein | Z-Image Turbo |
|--|----------------|--------------|---------------|
| Text encoders | CLIP-L + T5-XXL | Qwen3 (4B or 8B) | Qwen3 4B |
| CLIPLoader type | N/A (bundled) | `flux2` | `lumina2` |
| Latent node | EmptyLatentImage | EmptyFlux2LatentImage | EmptySD3LatentImage |
| Latent channels | 16 | 128 | 16 |
| Spatial downscale | 1/8 | 1/16 | 1/8 |
| Sampler | KSampler | SamplerCustomAdvanced | KSampler |
| Scheduler | simple | Flux2Scheduler | simple |
| Extra nodes | none | CFGGuider | ModelSamplingAuraFlow |

## Licensing Summary

**Commercial-safe (Apache 2.0):** Schnell, Klein 4B, Z-Image Turbo
**Non-commercial:** Klein 9B (need paid license from BFL for commercial use)

When running locally through ComfyUI, generated images contain no invisible watermarks or tracking. ComfyUI embeds workflow JSON in PNG metadata (strippable). No enforcement mechanism exists for locally-generated outputs.

## Sources

- FLUX.1 Schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- FLUX.2 Klein 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- FLUX.2 Klein 9B: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- Z-Image Turbo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Z-Image FP8: https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled
- ComfyUI examples: https://comfyanonymous.github.io/ComfyUI_examples/z_image/
