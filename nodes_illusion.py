import math

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.utils
from comfy_api.latest import ComfyExtension, io

LUMA = (0.2126, 0.7152, 0.0722)


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


def illusion_delta(x0, latent_format, gray, strength, blur, contrast, mask=None):
    # the preview factors map a latent pixel linearly to RGB, so luminance is a dot product with them
    factors = torch.tensor(latent_format.latent_rgb_factors, device=x0.device, dtype=torch.float32)
    luma = (factors @ torch.tensor(LUMA, device=x0.device)).to(x0.dtype)
    # smallest latent step that raises R, G and B by exactly 1, so brightening adds no colour cast
    neutral = (factors @ torch.linalg.solve(factors.T @ factors, torch.ones(3, device=x0.device))).to(x0.dtype)
    reshape = latent_format.latent_rgb_factors_reshape
    x = x0 if reshape is None else reshape(x0)
    lum = torch.einsum("bc...,c->b...", x, luma)
    h, w = lum.shape[-2:]
    frames = lum.shape
    lum = lum.reshape(-1, 1, h, w)

    radius = blur * max(h, w)
    target = comfy.utils.common_upscale(gray, w, h, "area", "center").to(x) * 2.0 - 1.0
    target = gaussian_blur(comfy.utils.repeat_to_batch_size(target, lum.shape[0]), radius)
    lum = gaussian_blur(lum, radius)
    target = lum.mean((2, 3), keepdim=True) + contrast * (target - target.mean((2, 3), keepdim=True))
    delta = (target - lum) * strength
    if mask is not None:
        delta = delta * comfy.utils.repeat_to_batch_size(comfy.utils.common_upscale(mask, w, h, "bilinear", "center").to(x), lum.shape[0])

    delta = delta.reshape(frames).unsqueeze(1) * neutral.view(-1, *[1] * (len(frames) - 1))
    if reshape is not None:
        # the only preview reshape is Flux2's pixel shuffle (128 channels -> 32 channels at 2x)
        delta = F.pixel_unshuffle(delta, h // x0.shape[-2])
    return delta


class IllusionGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IllusionGuidance",
            display_name="Illusion Guidance",
            category="model_patches",
            description="Hides a pattern in the generated image: while the composition forms, the blurred brightness of the predicted image is pulled toward the pattern, then the pull fades out and the model renders the details freely. Works like the QR Monster ControlNet on any model with latent previews.",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("pattern", tooltip="Black and white pattern, text, logo or QR code. White areas become bright, black areas dark."),
                io.Float.Input("strength", default=2.0, min=0.0, max=5.0, step=0.01, tooltip="How hard the early steps are pulled toward the pattern. About 2 for Qwen Image and Krea 2, 3 for Ideogram 4."),
                io.Float.Input("blur", default=0.02, min=0.0, max=0.25, step=0.001, tooltip="Size of the brightness blobs that must match, as a fraction of the image size. Lower keeps fine shapes such as QR modules (about 0.006), higher hides the pattern deeper."),
                io.Float.Input("contrast", default=0.8, min=0.0, max=2.0, step=0.01, tooltip="Brightness difference between the pattern's black and white areas. 1.0 maps them to black and white."),
                io.Boolean.Input("invert", default=False),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=0.8, min=0.0, max=1.0, step=0.001, tooltip="The pull fades out toward this point so the model renders the details on its own."),
                io.Mask.Input("mask", optional=True, tooltip="Only apply the illusion where the mask is white."),
            ],
            outputs=[io.Model.Output()],
            search_aliases=["qr monster", "illusion diffusion", "hidden image", "optical illusion", "hidden text"],
        )

    @classmethod
    def execute(cls, model, pattern, strength, blur, contrast, invert, start_percent, end_percent, mask=None) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(model)
        latent_format = model.get_model_object("latent_format")
        if latent_format.latent_rgb_factors is None:
            raise ValueError("Illusion Guidance needs a model with latent preview factors, {} has none.".format(type(latent_format).__name__))

        gray = (pattern[..., :3] @ torch.tensor(LUMA)).unsqueeze(1)
        if invert:
            gray = 1.0 - gray
        if mask is not None:
            mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)
        sigma_top = min(sigma_start, float(model_sampling.sigma_max))

        def illusion(args):
            x0 = args["denoised"]
            sigma = float(args["sigma"][0])
            if sigma > sigma_start or sigma <= sigma_end:
                return x0
            # push hard while the composition forms, then fade out so the model renders the details on its own
            fade = min(1.0, (sigma - sigma_end) / (sigma_top - sigma_end))
            return x0 + illusion_delta(x0, latent_format, gray, strength * fade, blur, contrast, mask)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(illusion)
        return io.NodeOutput(m)


class IllusionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IllusionGuidance,
        ]


async def comfy_entrypoint() -> IllusionExtension:
    return IllusionExtension()
