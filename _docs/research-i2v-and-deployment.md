# Research: I2V Best Practices & Cog-ComfyUI Deployment

Compiled 2026-03-25 from web research + codebase analysis of `/home/huyai/workspace/cog-comfyui/`.

---

## The "Frozen Then Jump" Problem

Our 5s i2v portrait held the pose for 3 seconds then jumped to a smile. This is a **known issue**.

**Root cause:** The model was trained on compressed video. It associates compression artifacts with motion. A clean, high-quality input image lacks those cues, so it freezes.

**Fixes (from real user reports):**

1. **Use the distilled model, not dev** — distilled produces motion 8/10 times vs dev frequently freezing ([GitHub #352](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/352))
2. **Add compression to source image** — set `LTXVPreprocess` `img_compression` to 35-42 (we used 18, too clean)
3. **Don't generate long clips in one pass** — generate 4-5s clips, extend with `LTXVExtendSampler`
4. **Lower the resolution** — lower res = more motion. Start at 512-640, upscale after
5. **Prompt for motion, not appearance** — describe what changes, not what exists
6. **Use two-stage pipeline** — half-res first, latent upscale second (this is what cog-comfyui does)

## Strength & CFG Settings

| Parameter | Portrait i2v | Longer clips (5s+) | Two-stage upscale |
|-----------|-------------|--------------------|--------------------|
| Image strength | 0.7 | 0.5-0.6 | 1.0 (lock frame) |
| CFG | 1.0 (distilled) | 1.0 | 1.0 |
| img_compression | 35-42 | 35-42 | N/A |
| Distilled LoRA strength | 0.5 | 0.5 | 0.2 |
| Steps (distilled) | 8 | 8 | 4 |

## Two-Stage Pipeline (from cog-comfyui)

This is how your colleague's deployment works:

**Stage 1 — Base generation at half resolution:**
```
width/2, height/2 → 8 steps → sigmas: 1.0...0.0
```

**Stage 2 — Latent upscale to full resolution:**
```
LTXVLatentUpsampler (2x) → 4 steps → sigmas: 0.85, 0.725, 0.4219, 0.0
```

This produces better temporal consistency than single-stage because:
- Stage 1 plans the motion at low res (cheaper, less VRAM)
- Stage 2 adds detail without re-planning motion

## First + Last Frame Conditioning

LTX 2.3 supports conditioning on both first AND last frame via `LTXVAddGuide`:
- First frame strength: 0.95-1.0 (lock the start)
- Last frame strength: 0.7-0.8 (slightly lower = more natural)
- Gives the model a motion **target** — instead of freezing, it interpolates

## Prompt Engineering for Portrait I2V

**Don't describe the image. Describe the motion.**

Bad: "A beautiful woman with blonde hair in a black top"
Good: "The subject slowly turns her head, a gentle breeze moves strands of hair across her face. She blinks naturally and gives a subtle smile."

Pattern:
1. Shot type ("medium close-up, shallow depth of field")
2. Motion description ("slowly turns", "hair sways", "blinks")
3. Camera ("static camera", or "slow dolly in")
4. Anti-freeze: prompt empty areas too ("background bokeh shifts subtly")

## Camera LoRAs

Available: dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, static

- Use at **strength 1.0** (designed for full strength)
- `static` LoRA helps stabilize portrait i2v
- `dolly_in` on portrait = natural "push toward subject" that drives motion

---

## Cog-ComfyUI Architecture

Source: `/home/huyai/workspace/cog-comfyui/`

### Key Design: No Server, No JSON

ComfyUI runs **in-process as a Python library**, not as a server:

```python
from nodes import NODE_CLASS_MAPPINGS

class Component:
    def _exec(self, node: str, **kwargs):
        node_class = NODE_CLASS_MAPPINGS[node]()
        func = getattr(node_class, node_class.FUNCTION)
        return func(**kwargs)
```

No HTTP API, no JSON workflow parsing, no server process management. Just direct Python calls to ComfyUI nodes.

### Component Pattern

```
Component (base class)
├── _download()         — lazy HuggingFace download, skip if exists
├── _exec()             — run a ComfyUI node directly
├── _preload_models()   — download weights + load into VRAM at init
└── run(**kwargs)       — execute generation pipeline
```

Each model family is a `Component` subclass: `LTX23`, `LTX23_D2V`, `Wan`, etc.

### LTX23 Specific (comfy_utils/ltx.py)

**Models downloaded at init:**
- `ltx-2.3-22b-dev-fp8.safetensors` (checkpoint)
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- `ltx-2.3-22b-distilled-lora-384.safetensors`
- `gemma_3_12B_it_fp4_mixed.safetensors`
- Abliterated Gemma LoRA

**I2V uses `LTXVImgToVideoInplace`** (not `LTXVImgToVideoConditionOnly` like we used):
- Encodes image directly into the latent with the VAE
- `bypass=True` for t2v mode, `bypass=False` for i2v

**Two-stage sampling:**
- Stage 1: `euler_ancestral_cfg_pp`, 9 sigma values, half resolution
- Stage 2: `euler_cfg_pp`, 4 sigma values, full resolution after `LTXVLatentUpsampler`

**Audio:** Generated in the same forward pass via `LTXVAudioVAEDecode`

### Production Optimizations

| Optimization | How |
|-------------|-----|
| TF32 matmul | `torch.backends.cuda.matmul.allow_tf32 = True` |
| Flash Attention | `torch.backends.cuda.enable_flash_sdp(True)` |
| Inference mode | `@torch.inference_mode()` on all generation |
| Model caching | Models loaded once at init, reused across requests |
| GPU warmup | Special seed triggers 0.25s matmul warmup for clock stabilization |
| Dimension rounding | `match_dimension_step(64)` for efficient tensor ops |
| Mixed precision | fp8 for UNETs, bf16 for LoRAs, fp4 for text encoder |

### Cog Build (cog.yaml)

- GPU: H100, CUDA 12.1
- Python 3.12, PyTorch 2.5.1
- 6 custom node packs: VideoHelperSuite, RES4LYF, ComfyUI-LTXVideo, KJNodes, Impact-Pack, LayerStyle
- Models downloaded at runtime, not bundled in Docker image

---

## Key Takeaways

1. **Our 5s i2v failed because**: single-stage, low compression (18), prompting for appearance not motion
2. **Fix**: two-stage pipeline + higher compression (35-42) + motion-focused prompts
3. **For Replicate deployment**: use in-process execution (no ComfyUI server), FP8 on H100, cache models across requests
4. **Node difference**: colleague uses `LTXVImgToVideoInplace`, we should test this vs `LTXVImgToVideoConditionOnly`

## Sources

- [GitHub: LTX-2 I2V frozen frame #11](https://github.com/Lightricks/LTX-2/issues/11)
- [GitHub: Static images with non-distilled #352](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/352)
- [Civitai: LTX I2V Tips](https://civitai.com/articles/11073/ltx-image-to-video-tips)
- [LTX 2.3 Prompt Guide](https://ltx.io/model/model-blog/ltx-2-3-prompt-guide)
- [ComfyUI-LTXVideo Looping Sampler](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md)
- [Official LTX 2.3 Workflows](https://docs.comfy.org/tutorials/video/ltx/ltx-2-3)
- Codebase: `/home/huyai/workspace/cog-comfyui/comfy_utils/ltx.py`
