import math

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_sampling
import comfy.patcher_extension
import comfy.utils
from comfy_api.latest import ComfyExtension, io

LUMA = (0.2126, 0.7152, 0.0722)
# Feature sizes (gaussian sigma as a fraction of the image) whose layout noise shaping writes into the starting noise.
NOISE_BANDS = (0.01, 0.02, 0.04, 0.08, 0.16)
# The latent step that brightens the decoded image by one unit, measured by encoding images and brightened copies through
# each VAE. The smallest step that raises the preview RGB is a different direction and decodes with grid artifacts.
BRIGHTER = {
    "Wan21": [
        0.4891, -0.1823, 0.7685, 0.9798, -0.7549, 0.4088, -0.6533, 0.0812,
        -0.7889, -0.1600, 0.6826, 0.1912, -0.1847, -0.8988, 0.7218, 0.6267,
    ],
    "QwenImage21": [
        -0.4691, -0.7159, 0.7543, 0.0815, -0.1990, 0.5420, -0.0718, 0.9820,
        -0.2594, -0.6647, 0.2700, 0.7785, -0.1208, 0.4479, 0.2205, 0.1623,
        -0.4163, 0.6137, -0.1774, 0.6412, -0.7036, -0.1117, -0.1468, 0.2326,
        -1.4445, 0.8007, 0.2623, 0.1252, 0.2054, -0.0653, -0.4782, 0.3244,
        0.1321, -0.0210, 2.5236, -0.1085, -0.4997, 0.1925, -0.1865, -0.4753,
        0.5521, 1.6951, 0.1884, 2.3339, 0.2819, 0.7836, -0.7801, -2.4338,
        -0.6756, 2.0795, -0.3174, -0.7613, -0.3016, -1.6615, -0.1960, 0.3489,
        0.2279, -2.7720, -0.1371, -0.5113, 0.1856, 0.5020, 0.2310, 0.1746,
    ],
}


def gaussian_blur(x, sigma):
    if sigma < 0.3:
        return x
    radius = math.ceil(sigma * 3)
    kernel = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    return F.conv2d(x, kernel.view(1, 1, -1, 1))


def luminance(x, latent_format):
    # the preview factors map a latent pixel linearly to RGB, so luminance is a dot product with them
    factors = torch.tensor(latent_format.latent_rgb_factors, device=x.device, dtype=torch.float32)
    if latent_format.latent_rgb_factors_reshape is not None:
        x = latent_format.latent_rgb_factors_reshape(x)
    return torch.einsum("bc...,c->b...", x, (factors @ torch.tensor(LUMA, device=x.device)).to(x.dtype))


def brighten(amount, latent_format, like):
    brighter = BRIGHTER.get(type(latent_format).__name__)
    if brighter is None:
        # uncalibrated latent: smallest step that raises R, G and B by exactly 1
        factors = torch.tensor(latent_format.latent_rgb_factors, device=like.device, dtype=torch.float32)
        brighter = factors @ torch.linalg.solve(factors.T @ factors, torch.ones(3, device=like.device))
    else:
        brighter = torch.tensor(brighter, device=like.device)
    delta = amount.unsqueeze(1) * brighter.to(like.dtype).view(-1, *[1] * (amount.ndim - 1))
    if delta.shape[-2] != like.shape[-2]:
        # the only preview reshape is Flux2's pixel shuffle (128 channels -> 32 channels at 2x)
        delta = F.pixel_unshuffle(delta, delta.shape[-2] // like.shape[-2])
    return delta


def fit(image, like, method="area"):
    h, w = like.shape[-2:]
    return comfy.utils.repeat_to_batch_size(comfy.utils.common_upscale(image, w, h, method, "center").to(like), like.shape[0])


def illusion_delta(x0, latent_format, gray, strength, blur, contrast, mask=None):
    lum = luminance(x0, latent_format)
    shape = lum.shape
    lum = lum.reshape(-1, 1, *shape[-2:])
    radius = blur * max(shape[-2:])
    target = gaussian_blur(fit(gray, lum) * 2.0 - 1.0, radius)
    lum = gaussian_blur(lum, radius)
    target = lum.mean((2, 3), keepdim=True) + contrast * (target - target.mean((2, 3), keepdim=True))
    delta = (target - lum) * strength
    if mask is not None:
        delta = delta * fit(mask, lum, "bilinear")
    return brighten(delta.reshape(shape), latent_format, x0)


def shape_noise(noise, latent_format, gray, amount, mask=None):
    # The model reads its composition from the low frequencies of the starting noise. Per feature size, swap a share of
    # the noise's own brightness layout for the pattern's at the noise's own energy, so the noise keeps the statistics
    # the model expects at every scale.
    lum = luminance(noise, latent_format)
    shape = lum.shape
    lum = lum.reshape(-1, 1, *shape[-2:])
    target = fit(gray, lum) * 2.0 - 1.0
    size = max(shape[-2:])
    noise_blurs = [gaussian_blur(lum, s * size) for s in NOISE_BANDS]
    pattern_blurs = [gaussian_blur(target, s * size) for s in NOISE_BANDS]
    noise_blurs.append(noise_blurs[-1].mean((2, 3), keepdim=True))
    pattern_blurs.append(pattern_blurs[-1].mean((2, 3), keepdim=True))
    delta = torch.zeros_like(lum)
    for k in range(len(NOISE_BANDS)):
        noise_band = noise_blurs[k] - noise_blurs[k + 1]
        pattern_band = pattern_blurs[k] - pattern_blurs[k + 1]
        gain = noise_band.std((2, 3), keepdim=True) / (pattern_band.std((2, 3), keepdim=True) + 1e-6)
        delta = delta + (math.sqrt(1.0 - amount * amount) - 1.0) * noise_band + amount * gain * pattern_band
    if mask is not None:
        delta = delta * fit(mask, lum, "bilinear")
    return noise + brighten(delta.reshape(shape), latent_format, noise)


class IllusionGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IllusionGuidance",
            display_name="Illusion Guidance",
            category="model_patches",
            description="Hides a pattern in the generated image. While the composition forms, the blurred brightness of the predicted image is pulled toward the pattern, so the model builds the scene around it; the pull then fades out and the model renders the details freely. Works like the QR Monster ControlNet on any model with latent previews.",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("pattern", tooltip="Black and white pattern, text, logo or QR code. White areas become bright, black areas dark."),
                io.Float.Input("strength", default=3.0, min=0.0, max=10.0, step=0.01, tooltip="How hard the early steps are pulled toward the pattern."),
                io.Float.Input("contrast", default=0.5, min=0.0, max=2.0, step=0.01, tooltip="Brightness difference asked between the pattern's black and white areas. 1.0 asks for black and white, which no natural scene can supply; 0.35 to 0.5 lets the model use real content."),
                io.Float.Input("blur", default=0.02, min=0.0, max=0.25, step=0.001, tooltip="Size of the brightness blobs that must match, as a fraction of the image size. Lower keeps fine shapes such as QR modules (about 0.006)."),
                io.Boolean.Input("invert", default=False),
                io.Float.Input("fade_start", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="Noise level (1 = pure noise, 0 = finished image) down to which the pull stays at full strength."),
                io.Float.Input("fade_end", default=0.6, min=0.0, max=1.0, step=0.01, tooltip="Noise level at which the pull has faded to zero. Pulling at lower noise levels paints the pattern over the scene instead of building the scene from it."),
                io.Float.Input("noise_shaping", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Share of the starting noise's brightness layout replaced by the pattern's. Helps models that fix their composition in the first step (Qwen Image 2.1: about 0.5). Too much makes the model draw the pattern as a flat graphic."),
                io.Mask.Input("mask", optional=True, tooltip="Only apply the illusion where the mask is white."),
            ],
            outputs=[io.Model.Output()],
            search_aliases=["qr monster", "illusion diffusion", "hidden image", "optical illusion", "hidden text"],
        )

    @classmethod
    def execute(cls, model, pattern, strength, contrast, blur, invert, fade_start, fade_end, noise_shaping, mask=None) -> io.NodeOutput:
        if strength == 0 and noise_shaping == 0:
            return io.NodeOutput(model)
        latent_format = model.get_model_object("latent_format")
        if latent_format.latent_rgb_factors is None:
            raise ValueError("Illusion Guidance needs a model with latent preview factors, {} has none.".format(type(latent_format).__name__))

        gray = (pattern[..., :3] @ torch.tensor(LUMA)).unsqueeze(1)
        if invert:
            gray = 1.0 - gray
        if mask is not None:
            mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
        flow = isinstance(model.get_model_object("model_sampling"), comfy.model_sampling.CONST)

        def illusion(args):
            x0 = args["denoised"]
            sigma = float(args["sigma"][0])
            # share of noise in what the model is looking at: flow models mix (1 - sigma) image with sigma noise, others add sigma noise
            noise_level = sigma if flow else sigma / (1.0 + sigma)
            fade = min(1.0, (noise_level - fade_end) / max(fade_start - fade_end, 1e-6))
            if fade <= 0:
                return x0
            return x0 + illusion_delta(x0, latent_format, gray, strength * fade, blur, contrast, mask)

        def shaped_noise(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar):
            return executor(guider, sigmas, extra_args, callback, shape_noise(noise, latent_format, gray, noise_shaping, mask), latent_image, denoise_mask, disable_pbar)

        m = model.clone()
        if strength > 0:
            m.set_model_sampler_post_cfg_function(illusion)
        if noise_shaping > 0:
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "illusion_noise_shaping", shaped_noise)
        return io.NodeOutput(m)


class IllusionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IllusionGuidance,
        ]


async def comfy_entrypoint() -> IllusionExtension:
    return IllusionExtension()
