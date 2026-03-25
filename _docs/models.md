# Model Inventory

Check before downloading. Run `du -sh models/*/` to see current sizes.

## Currently Installed

### Shared (used by all workflows)

| Model | Size | Path | Purpose |
|-------|------|------|---------|
| `gemma_3_12B_it_fp4_mixed.safetensors` | 8.8 GB | `text_encoders/` | Text encoder (Gemma 3 12B, fp4) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.1 GB | `loras/` | Distilled LoRA — 8-step generation |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | `latent_upscale_models/` | 2x latent upscale (two-stage pipeline) |

### Path A: GGUF

| Model | Size | Path | Workflows |
|-------|------|------|-----------|
| `ltx-2.3-22b-dev-Q4_K_M.gguf` | 14 GB | `unet/` | `eugene_ltx23_gguf_t2v.json`, `eugene_ltx23_two_stage_t2v.json` |
| `ltx-2.3-22b-dev_video_vae.safetensors` | 1.4 GB | `vae/` | Same (GGUF needs separate VAE) |
| `ltx-2.3-22b-dev_embeddings_connectors.safetensors` | 2.2 GB | `checkpoints/` | Same (GGUF needs separate connectors) |

### Path B: FP8

| Model | Size | Path | Workflows |
|-------|------|------|-----------|
| `ltx-2.3-22b-dev-fp8.safetensors` | 28 GB | `checkpoints/` | `eugene_ltx23_fp8_t2v.json`, `eugene_ltx23_fp8_two_stage_t2v.json` |
| `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 600 MB | `loras/` | Same (uncensors Gemma text encoder) |

Note: FP8 checkpoint bundles its own VAE and connectors — no separate files needed.

---

## Text-to-Image Models

See `_docs/t2i-models.md` for full details, node chains, and architecture differences.

### Shared (T2I)

| Model | Size | Path | Used By |
|-------|------|------|---------|
| `qwen_3_4b.safetensors` | 7.5 GB | `text_encoders/` | Klein 4B, Z-Image Turbo (type=flux2 / lumina2) |
| `flux2-vae.safetensors` | 321 MB | `vae/` | Klein 4B |
| `ae.safetensors` | 320 MB | `vae/` | Z-Image Turbo |

### FLUX.2 Klein 4B

| Model | Size | Path | Workflows |
|-------|------|------|-----------|
| `flux-2-klein-4b.safetensors` | 7.2 GB | `diffusion_models/` | `eugene_flux2_klein4b_bf16_t2i.json` |
| `flux-2-klein-4b-fp8.safetensors` | 3.8 GB | `diffusion_models/` | `eugene_flux2_klein4b_fp8_t2i.json` |

### Z-Image Turbo

| Model | Size | Path | Workflows |
|-------|------|------|-----------|
| `z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors` | 5.7 GB | `diffusion_models/` | `eugene_zimage_turbo_fp8_t2i.json` |

## Storage

- **Current usage:** ~107 GB
- **Limit:** 300 GB
- Video models added by: eugene, 2026-03-24
- T2I models added by: eugene, 2026-03-25
