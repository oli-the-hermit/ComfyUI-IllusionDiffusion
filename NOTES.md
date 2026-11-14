# Lab notebook

Experiments toward v2, on Krea 2 Turbo (8 steps) and Qwen Image 2.1 (25 steps), 1024 px. One entry per experiment: what was tested, the numbers, the verdict. Everything here ran in a dev copy of the node; nothing below is in `nodes_illusion.py` yet unless an entry says so.

## How results are scored
A quick set (2 prompts × spiral, QR, knight silhouette × 2 seeds = 12 images per model), each compared with the same seed without guidance.

- **Reveal:** correlation between the pattern and the image as seen from far away (64 px, averaged in linear light).
- **Natural:** texture and colour kept inside the pattern's dark and bright areas, compared with the unguided image, minus clipped black or white pixels. Too much texture or colour counts against an image as much as too little.
- **Score:** reveal × natural. A deliberately painted run scores 0.07; v1 scores 0.34 (Krea) and 0.28 (Qwen).

The score is a filter, not a judge. It was fooled once (E7), so every kept change is also checked by eye.

## E1. When does each model decide its layout?
Logged the predicted image at every step of 8 unguided runs per model and compared each feature size with the final image.

- Qwen fixes its composition almost at once: after 1 of 25 steps, coarse layout already matches the final image at 0.93 and spiral-sized features at 0.67.
- Krea decides more gradually, over its first 4 of 8 steps.

Verdict: the two models do not share one timing rule. On Qwen the layout is read from the starting noise before a push on the prediction can register.

## E2 and E3. One push at a single step
E3 gave every step the same displacement of the noisy image (0.4), so steps can be compared fairly.

| Noise level of the push | Krea reveal / natural | Qwen reveal / natural |
|---|---|---|
| Above 0.9 | 0.62 / 0.08–0.32 | 0.85 / 0.09–0.26 |
| About 0.8 | 0.75 / 0.63 | 0.86 / 0.58 |
| 0.75–0.65 | 0.81 / 0.81 | 0.80 / 0.79 |
| About 0.5 | 0.55 / 0.88 | 0.70 / 0.88 |
| Below 0.35 | 0.39 / 0.94 | 0.49–0.58 / 0.92 |

Verdict: three regimes, the same on both models.
1. **Too early (noise above about 0.9):** a strong push changes what kind of image it is. The model draws a flat black-and-white graphic.
2. **The window (about 0.8 down to 0.55):** the image type is settled but content is still open, so the push becomes content or lighting.
3. **Late:** the push stays as an overlay at whatever size it was applied.

V1 fades from the start, so most of its strength lands in regime 1. That is the haze on Qwen.

## E4 and E5. Seeding the starting noise
E4 added the pattern to the noise's brightness. E5 instead swaps a share `rho` of the noise's own layout for the pattern's at every feature size, keeping the noise's energy unchanged.

- E4: striking compositions at the right amplitude (a village with spiral paths on Krea), but a narrow range. A little more and both models draw a literal graphic. Large flat pattern areas trigger it first. Dropped.
- E5: no cliff on Qwen (reveal 0.16 → 0.45 as rho goes 0.5 → 1.0, natural stays above 0.7). Krea ignores it up to 0.7 and collapses at 1.0.

Verdict: the matched seed is the only thing so far that makes Qwen build structure from the pattern (houses lined up along spiral arms). Useful on Qwen at low rho, not on Krea.

## E6. Push only inside the window
Full push between noise 0.8 and 0.6, ramping in from 0.9 and out to 0.45.

- Strength 1.2: Krea 0.43, Qwen 0.47 (v1: 0.34, 0.28).
- Result by eye: a clean light-and-shadow overlay on an unchanged scene. V1 on Krea restructures the scene instead (roads follow the spiral, a forest takes the knight's shape), which is the more striking look.

Verdict: early pushes give structure, window pushes give shading. Both are wanted; the window alone is not the answer.

## E7. Brighten by scaling (dropped)
Scaled the whole latent to brighten, the way exposure does, to stop bright areas washing out. The first version of the metric rated it best. The images had neon colours and a dot-grid texture. Dropped, and the metric was made two-sided because of it.

## E8. Which direction is "brighter" in the latent?
V1 pushes along the smallest latent step that raises R, G and B equally. Measured the direction a real brightness change moves the latent along, by encoding images and brightened copies through each VAE.

- The two directions differ a lot: cosine 0.71 (Krea's VAE), 0.62 (Qwen's).
- Decoding the same smooth brightness change along each: the v1 direction leaves a fish-scale grid on Krea and a blue tint on Qwen. The measured direction gives clean, soft light and shadow.

Verdict: keep. This is the source of the dot texture in v1 images. The measured direction is also about 40–60% stronger, so strength and contrast need retuning (E9). Calibration so far used 6 images per model and needs redoing on a larger set before it ships.

## E9. Retuned strength and contrast with the measured direction
V1 schedule, measured direction.

| Config | Krea reveal / natural / score | Qwen reveal / natural / score |
|---|---|---|
| v1 (strength 2, contrast 0.8) | 0.92 / 0.38 / 0.34 | 0.95 / 0.30 / 0.28 |
| strength 2, contrast 0.8 | 0.96 / 0.18 / 0.18 | |
| strength 1.5, contrast 0.5 | 0.91 / 0.41 / 0.37 | 0.89 / 0.52 / 0.46 |
| strength 1.5, contrast 0.35 | 0.82 / 0.59 / 0.48 | |
| strength 1.0, contrast 0.5 | 0.70 / 0.66 / 0.46 | |

Verdict: contrast 0.8 asks for near-black and near-white areas, which nothing natural can supply. At contrast 0.5 the knight on Krea is a lit pine forest with individual trees, and Qwen loses its haze. By eye these are the best images so far on both models.

## E10. Seed plus guidance on Qwen
Measured direction, contrast 0.4–0.5.

- rho 1.0 or 0.7 with strength 1.5: collapses to a graphic (Qwen draws a chess-piece icon).
- rho 0.5 with strength 1.5: real structure, but the dark arms go black (score 0.29).
- rho 0.35 with strength 1.0, contrast 0.4: reveal 0.86, natural 0.55, score 0.46. By eye the most natural Qwen result.
- Krea with rho 0.6: collapse (score 0.10). No seed on Krea.

## Current best recipes (quick set only, not yet confirmed on the full set)
- **Krea 2:** measured direction, strength 1.5, contrast 0.5, v1 schedule.
- **Qwen 2.1:** measured direction, strength 1.0–1.5, contrast 0.4–0.5, v1 schedule, optional seed at rho 0.35.

## Open
- Confirm on the full set (6 prompts × 5 patterns × 2 seeds).
- Recalibrate the measured direction on more images.
- QR scans are down at these gentler settings (v1 scanned 50% on the quick set). QR needs its own stronger setting or the fine-detail work from the plan.
- Combine early structure with window shading on Krea; band-wise timing; margin targets; samplers; regional prompts.
