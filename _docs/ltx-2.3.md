# LTX 2.3

22B parameter video generation model from Lightricks. Released March 2026.

Supports: text-to-video, image-to-video, audio+video, 4K, portrait/landscape, camera controls.

## What We Run

Two paths set up on this machine:

### Path A: GGUF Q4_K_M (default)

Best for 24 GB VRAM. Comfortable headroom.

| File | Size | Location |
|------|------|----------|
| `ltx-2.3-22b-dev-Q4_K_M.gguf` | 14.3 GB | `models/unet/` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | 9.45 GB | `models/text_encoders/` |
| `ltx-2.3-22b-dev_embeddings_connectors.safetensors` | 2.31 GB | `models/checkpoints/` |
| `ltx-2.3-22b-dev_video_vae.safetensors` | 1.45 GB | `models/vae/` |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.61 GB | `models/loras/` |

Requires `ComfyUI-GGUF` custom node.

**Verified working workflow:** `workflows/eugene_ltx23_gguf_t2v.json`

GGUF node chain (differs from official Lightricks examples which use `CheckpointLoaderSimple`):

```
UnetLoaderGGUF ─────→ LoraLoaderModelOnly ─→ CFGGuider ─→ SamplerCustomAdvanced ─→ VAEDecodeTiled ─→ CreateVideo ─→ SaveVideo
VAELoader ──────────────────────────────────────────────────────────────────────────→ ↑
LTXAVTextEncoderLoader ─→ CLIPTextEncode (pos) ─→ LTXVConditioning ─→ CFGGuider     │
                        └→ CLIPTextEncode (neg) ─→ ↑                                │
EmptyLTXVLatentVideo ───────────────────────────────────────→ SamplerCustomAdvanced  │
ManualSigmas ───────────────────────────────────────────────→ ↑                      │
KSamplerSelect ─────────────────────────────────────────────→ ↑                      │
RandomNoise ────────────────────────────────────────────────→ ↑                      │
```

Key differences from official examples:
- `UnetLoaderGGUF` replaces `CheckpointLoaderSimple` (GGUF doesn't bundle VAE)
- Separate `VAELoader` needed for `ltx-2.3-22b-dev_video_vae.safetensors`
- `LTXAVTextEncoderLoader` uses `ckpt_name` to find the embeddings connectors in `models/checkpoints/`
- `LTXVConditioning` takes only (positive, negative, frame_rate) — no latent input
- Distilled LoRA at strength 0.5 gives 8-step generation (via ManualSigmas)

### Path B: FP8 (higher quality, tight on VRAM)

| File | Size | Location |
|------|------|----------|
| `ltx-2.3-22b-dev-fp8.safetensors` | 29.1 GB | `models/checkpoints/` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | 9.45 GB | `models/text_encoders/` |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.61 GB | `models/loras/` |
| `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 628 MB | `models/loras/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 996 MB | `models/latent_upscale_models/` |

No custom nodes needed. Uses `CheckpointLoaderSimple` (standard ComfyUI path). Launch with `comfyui` — same as Path A.

## Generation Constraints

- **Width/Height:** must be divisible by 32 (e.g. 1280x720, 1024x576)
- **Frame count:** must be (8n + 1) — valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
- **Prompts:** use long, descriptive prompts for best results
- **Camera controls:** dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, static, focus_shift

## Two-Stage Pipeline (recommended)

**Workflow:** `workflows/eugene_ltx23_two_stage_t2v.json`

Generates at half resolution, then upscales in latent space for detail. Better temporal consistency than single-stage, especially for 5s+ clips.

**Stage 1** — motion planning at half res:
- `EmptyLTXVLatentVideo` at `width/2, height/2`
- 8 steps, `euler_ancestral_cfg_pp`, LoRA strength 0.5
- For i2v: add `LoadImage → LTXVPreprocess(compression=38) → ImageScale → LTXVImgToVideoInplace(strength=0.5-0.7)` before the sampler

**Stage 2** — detail refinement at full res:
- `LTXVLatentUpsampler` (2x, uses `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`)
- 4 steps, `euler_cfg_pp`, LoRA strength 0.2
- Sigmas: `0.85, 0.725, 0.4219, 0.0`

**I2V tips:**
- Use `LTXVImgToVideoInplace` (not `LTXVImgToVideoConditionOnly`)
- `img_compression`: 35-42 (model needs compression artifacts to drive motion)
- Strength 0.5-0.6 for 5s+ clips, 0.7 for short clips
- Prompt for **motion**, not appearance

## Custom Nodes Required

| Node | Purpose |
|------|---------|
| `ComfyUI-GGUF` | Loading GGUF quantized models (Path A) |
| `ComfyUI-LTXVideo` | Advanced LTX 2.3 features, IC-LoRA, low-VRAM loaders |
| `ComfyUI-VideoHelperSuite` | Video output/preview |

## Sources

- Model: https://huggingface.co/Lightricks/LTX-2.3
- GGUF: https://huggingface.co/unsloth/LTX-2.3-GGUF
- FP8: https://huggingface.co/Lightricks/LTX-2.3-fp8
- Text encoder: https://huggingface.co/Comfy-Org/ltx-2/tree/main/split_files/text_encoders
- Workflows: https://docs.comfy.org/tutorials/video/ltx/ltx-2-3
- ComfyUI-LTXVideo: https://github.com/Lightricks/ComfyUI-LTXVideo
