from __future__ import annotations

import base64
import io
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn

from autoencoder_lab.evaluation import reconstruction_metrics
from autoencoder_lab.protocols import EncoderDecoder


def image_data_url(image: Tensor | NDArray[np.float32]) -> str:
    values = image.detach().cpu().numpy() if isinstance(image, Tensor) else image
    values = np.asarray(values, dtype=np.float32).squeeze()
    pixels = np.clip(values * 255, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").resize((224, 224), Image.Resampling.NEAREST).save(
        buffer,
        format="PNG",
    )
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_upload(content: bytes) -> Tensor:
    if len(content) > 1024 * 1024:
        raise ValueError("upload exceeds the 1 MB limit")
    try:
        candidate = Image.open(io.BytesIO(content))
        candidate.verify()
        image = Image.open(io.BytesIO(content)).convert("L").resize(
            (28, 28),
            Image.Resampling.BILINEAR,
        )
    except Exception as error:
        raise ValueError("upload is not a valid image") from error
    values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).reshape(1, 1, 28, 28)


def reconstruct(model: nn.Module, image: Tensor) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        autoencoder = cast(EncoderDecoder, model)
        latent = autoencoder.encode(image)
        prediction = autoencoder.decode(latent)
    error = torch.abs(prediction - image)
    return {
        "original": image_data_url(image[0]),
        "reconstruction": image_data_url(prediction[0]),
        "absolute_error": image_data_url(error[0]),
        "latent": [round(float(value), 6) for value in latent[0].tolist()],
        "metrics": reconstruction_metrics(image, prediction).to_dict(),
    }
