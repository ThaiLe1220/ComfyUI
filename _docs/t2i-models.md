# Text-to-Image Models

2 models for T2I on RTX 3090 (24 GB VRAM). Both Apache 2.0, 4 steps, distilled.

## Quick Comparison

| Model | Params | Precision | VRAM | Workflow |
|-------|--------|-----------|------|----------|
| FLUX.2 [klein] 4B | 4B | BF16 / FP8 | ~16 GB / ~11 GB | `eugene_flux2_klein4b_bf16_t2i.json`, `eugene_flux2_klein4b_fp8_t2i.json` |
| Z-Image Turbo | 6B | FP8 | ~14.5 GB | `eugene_zimage_turbo_fp8_t2i.json` |

## FLUX.2 [klein] 4B

By Black Forest Labs. 4B params, step-distilled. Apache 2.0.

| File | Size | Path |
|------|------|------|
| `flux-2-klein-4b.safetensors` | 7.2 GB | `diffusion_models/` |
| `flux-2-klein-4b-fp8.safetensors` | 3.8 GB | `diffusion_models/` |
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

Settings: steps=4, cfg=1.0, sampler=euler, Flux2Scheduler. BF16 and FP8 produce identical speed on RTX 3090 (no FP8 tensor cores). FP8 uses less VRAM.

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

**Critical:** CLIPLoader type must be `lumina2` (not `flux2`). Same `qwen_3_4b.safetensors` file, different tokenizer.

## Architecture Differences

| | FLUX.2 Klein 4B | Z-Image Turbo |
|--|----------------|---------------|
| Text encoder | Qwen3 4B | Qwen3 4B |
| CLIPLoader type | `flux2` | `lumina2` |
| Latent node | EmptyFlux2LatentImage | EmptySD3LatentImage |
| Latent channels | 128 | 16 |
| Spatial downscale | 1/16 | 1/8 |
| Sampler | SamplerCustomAdvanced | KSampler |
| Scheduler | Flux2Scheduler | simple |
| Extra nodes | CFGGuider | ModelSamplingAuraFlow |

## Sources

- FLUX.2 Klein 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Z-Image Turbo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Z-Image FP8: https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled
