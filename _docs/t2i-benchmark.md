# T2I Model Benchmark

Tested 2026-03-25. Both models: 4 steps, CFG 1.0, Apache 2.0.

## Replicate H100 (measured, 54 predictions per model)

| Model | Avg | Min | Max | $/image | $/MP | Images/$1 |
|-------|-----|-----|-----|---------|------|-----------|
| **Klein 4B** | **0.86s** | 0.64s | 1.14s | **$0.00131** | **$0.00124** | **~760** |
| Z-Image Turbo | 1.03s | 0.80s | 1.30s | $0.00157 | $0.00150 | ~635 |

Rate: $0.001525/sec ($5.49/hr) from [replicate.com/pricing](https://replicate.com/pricing). Times are `predict_time` from Replicate API (GPU inference only, excludes cold boot).

### By resolution

| Model | 1024×1024 (1.05MP) | 832×1248 (1.04MP) | 1248×832 (1.04MP) |
|-------|-------------------|-------------------|-------------------|
| Klein 4B | 0.85s / $0.00130 | 0.85s / $0.00129 | 0.88s / $0.00135 |
| Z-Image | 1.01s / $0.00154 | 1.06s / $0.00162 | 1.03s / $0.00157 |

Cost is consistent across aspect ratios at the same megapixel count.

## RTX 3090 (measured, 120 images)

| Resolution | Z-Image Turbo | Klein 4B | Klein advantage |
|-----------|--------------|----------|-----------------|
| 1024×1024 | 5.01s | **3.52s** | 30% faster |
| 832×1248 | 5.01s | **3.27s** | 35% faster |
| 1248×832 | 5.01s | **3.26s** | 35% faster |
| 1536×1024 | 7.07s | **4.80s** | 32% faster |

## Comparison with managed APIs

| Service | $/image (1MP) | Images/$1 | vs Klein self-hosted |
|---------|---------------|-----------|---------------------|
| **Klein 4B (self-hosted H100)** | **$0.00131** | **760** | — |
| Z-Image (self-hosted H100) | $0.00157 | 635 | 1.2x more expensive |
| Cloudflare Klein 4B | $0.00138 | 723 | ~same |
| Replicate Flux Dev (managed) | $0.025 | 40 | 19x more expensive |
| Replicate Flux Pro (managed) | $0.04 | 25 | 31x more expensive |

## Deployed models

- Klein 4B: https://replicate.com/enhancerandroid02/eu_flux_1
- Z-Image Turbo: https://replicate.com/enhancerandroid02/eu_z_1

## Test data

18 prompts × 3 resolutions × 2 models = 108 images tested on Replicate H100.

Composites: `output/replicate_comparison/composites/{01-18}_{name}_{resolution}_{ratio}.png`
Raw images: `output/replicate_comparison/`
Timing data: `output/replicate_comparison/results.json`
Prompts: `output/replicate_comparison/prompts.txt`
