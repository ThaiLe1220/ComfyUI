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

## Storage

- **Current usage:** ~63 GB
- **Limit:** 300 GB
- Added by: eugene, 2026-03-24
