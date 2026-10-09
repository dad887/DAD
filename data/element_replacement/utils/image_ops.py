from __future__ import annotations

from PIL import Image, ImageFilter

from utils.masking import trim_alpha


def fit_asset(
    img: Image.Image, box: tuple[int, int, int, int], rotation: float = 0.0
) -> tuple[Image.Image, tuple[int, int]]:
    img = trim_alpha(img.convert("RGBA"))
    (x0, y0, x1, y1) = box
    (bw, bh) = (max(1, x1 - x0), max(1, y1 - y0))
    scale = min(bw / img.width, bh / img.height)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    if abs(rotation) > 0.01:
        img = img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        img = trim_alpha(img)
    return (img, (int(x0 + (bw - img.width) / 2), int(y0 + (bh - img.height) / 2)))


def add_shadow(canvas: Image.Image, layer: Image.Image, xy: tuple[int, int]) -> None:
    blur = 16
    opacity = 58
    offset = (0, 10)
    pad = blur * 3 + max(abs(offset[0]), abs(offset[1]))
    alpha = layer.getchannel("A")
    shadow_alpha = Image.new("L", (layer.width + pad * 2, layer.height + pad * 2), 0)
    shadow_alpha.paste(alpha.point(lambda p: int(p * opacity / 255)), (pad, pad))
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(blur))
    shadow = Image.new("RGBA", shadow_alpha.size, (10, 18, 30, 0))
    shadow.putalpha(shadow_alpha)
    canvas.alpha_composite(shadow, (xy[0] + offset[0] - pad, xy[1] + offset[1] - pad))
