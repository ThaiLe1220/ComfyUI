# ComfyUI Workspace

## Quick Start

```bash
comfyui
```

That's it. Activates the conda env and launches with Manager + previews enabled.

Access at: http://127.0.0.1:8188

## Machine Specs

- **GPU:** RTX 3090 (24 GB VRAM)
- **CPU:** Intel i3-12100F (4c/8t)
- **RAM:** 32 GB
- **Storage:** 1 TB NVMe
- **CUDA:** 12.4 toolkit, driver supports up to 13.0
- **Conda env:** `comfyui` (Python 3.12, PyTorch 2.11 + cu130)

## VRAM Limits

24 GB VRAM means:
- Use **GGUF quantized** or **FP8** models, not full BF16
- Enable **tiled VAE decode** for video
- Text encoder gets offloaded to CPU automatically
- Keep video resolution reasonable (720p for generation, upscale after)

## Workflows

Save in `workflows/` with a prefix:

```
workflows/
├── eugene_ltx23_t2v.json
├── khoa_wan22_i2v.json
└── thanh_flux_img.json
```

## Custom Nodes

Install through the **Manager UI** (button in top-right of ComfyUI). Don't clone manually.

## House Rules

1. **Check `models/` before downloading** — someone may already have it
2. **Check `_docs/models.md`** for what's installed and sizes
3. **Don't leave ComfyUI running idle** — it holds VRAM
4. **Keep total models under 300 GB** — we need headroom
