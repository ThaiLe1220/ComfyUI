# T2I Model Benchmark

Tested 2026-03-25. Distilled models, 4 steps, CFG 1.0, 1024×1024.

## Speed (RTX 3090, measured)

10 prompts × 4 resolutions = 120 images.

| Resolution | Z-Image Turbo 6B | Klein 4B | Klein advantage |
|-----------|-----------------|----------|-----------------|
| 1024×1024 | 5.01s | **3.52s** | 30% faster |
| 832×1248 | 5.01s | **3.27s** | 35% faster |
| 1248×832 | 5.01s | **3.26s** | 35% faster |
| 1536×1024 | 7.07s | **4.80s** | 32% faster |

## Cost

| Platform | Klein 4B (1024²) | Source |
|----------|-----------------|--------|
| Cloudflare Workers AI | 723 img/$1 ($0.00138/img) | [cloudflare.com](https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/) |
| Replicate H100 ($0.001525/s) | not yet benchmarked | [replicate.com/pricing](https://replicate.com/pricing) |

Cloudflare pricing is per-tile (serverless), not GPU-time. Z-Image Turbo is not available on Cloudflare.

Both models are **Apache 2.0** (commercial OK). Neither has watermarks or tracking when run locally.

## Summary

Klein 4B is 30-35% faster than Z-Image Turbo on RTX 3090 across all resolutions. Quality is subjective — compare the composites side by side. Klein is 4B params, Z-Image is 6B.

Composite images: `output/t2i_comparison/{01-10}_{name}_{resolution}_{ratio}.png`
Prompts used: `output/t2i_comparison/prompts.txt`
