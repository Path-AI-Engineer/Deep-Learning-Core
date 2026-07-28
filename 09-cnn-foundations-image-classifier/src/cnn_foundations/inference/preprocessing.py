from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg"}


@dataclass(frozen=True)
class ProcessedImage:
    tensor: torch.Tensor
    preview_data_url: str
    original_size: tuple[int, int]
    mode: str
    preprocessing_summary: tuple[str, ...]


def _preview(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def process_image_bytes(
    content: bytes,
    *,
    mime_type: str,
    mean: float,
    std: float,
) -> ProcessedImage:
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise TypeError("only PNG and JPEG images are supported.")
    if not content:
        raise ValueError("uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise OverflowError("uploaded image exceeds the 2 MiB limit.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            original_size = source.size
            mode = source.mode
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("uploaded image dimensions exceed the pixel limit.")
            grayscale = source.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("uploaded content is not a decodable image.") from error
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    tensor = torch.from_numpy((array - mean) / std).unsqueeze(0).unsqueeze(0)
    return ProcessedImage(
        tensor=tensor,
        preview_data_url=_preview(grayscale),
        original_size=original_size,
        mode=mode,
        preprocessing_summary=(
            f"Decoded {mime_type} in memory",
            "Converted to one grayscale channel",
            "Resized to 28 x 28 pixels",
            f"Scaled to [0, 1] and normalized with mean={mean:.6f}, std={std:.6f}",
            "Produced NCHW tensor [1, 1, 28, 28]",
        ),
    )


def tensor_to_preview(tensor: torch.Tensor, *, mean: float, std: float) -> str:
    value = tensor.detach().cpu().squeeze().numpy()
    restored = np.clip(value * std + mean, 0.0, 1.0)
    image = Image.fromarray(np.uint8(restored * 255), mode="L")
    return _preview(image)

