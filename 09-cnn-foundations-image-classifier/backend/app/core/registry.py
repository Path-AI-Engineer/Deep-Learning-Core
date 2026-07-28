from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from cnn_foundations.artifacts.bundle import BundleContents, load_bundle
from cnn_foundations.data.gallery import SampleGallery
from cnn_foundations.explanations.activations import ActivationInspector
from cnn_foundations.inference.predictor import Predictor
from cnn_foundations.inference.preprocessing import ProcessedImage, process_image_bytes


class ModelRegistry:
    def __init__(
        self,
        *,
        bundle_path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.bundle_path = bundle_path or Path(
            os.getenv("CNN_BUNDLE_PATH", "artifacts/models/cnn/v1.0.0")
        )
        self.data_root = data_root or Path(
            os.getenv("FASHION_MNIST_ROOT", "data/raw")
        )
        self.bundle: BundleContents | None = None
        self.predictor: Predictor | None = None
        self.inspector: ActivationInspector | None = None
        self.gallery: SampleGallery | None = None
        self.model_error: str | None = None
        self.gallery_error: str | None = None

    def load(self) -> None:
        try:
            self.bundle = load_bundle(self.bundle_path)
            self.predictor = Predictor(self.bundle)
            self.inspector = ActivationInspector(self.bundle.model)
        except (FileNotFoundError, ValueError, OSError) as error:
            self.model_error = str(error)
        try:
            self.gallery = SampleGallery(self.data_root)
        except (RuntimeError, OSError) as error:
            self.gallery_error = str(error)

    @property
    def ready(self) -> bool:
        return self.predictor is not None and self.gallery is not None

    def require_bundle(self) -> BundleContents:
        if self.bundle is None:
            raise RuntimeError(
                "The approved FashionMNIST CNN bundle is not available."
            )
        return self.bundle

    def require_predictor(self) -> Predictor:
        if self.predictor is None:
            raise RuntimeError(
                "The approved FashionMNIST CNN bundle is not available."
            )
        return self.predictor

    def require_gallery(self) -> SampleGallery:
        if self.gallery is None:
            raise RuntimeError(
                "The official FashionMNIST test gallery is not available."
            )
        return self.gallery

    def preprocessing(self, content: bytes, mime_type: str) -> ProcessedImage:
        bundle = self.require_bundle()
        contract: dict[str, Any] = bundle.files["preprocessing.json"]
        return process_image_bytes(
            content,
            mime_type=mime_type,
            mean=float(contract["mean"]),
            std=float(contract["std"]),
        )

    def process_sample(self, sample_id: str) -> tuple[dict[str, Any], ProcessedImage]:
        gallery = self.require_gallery()
        sample, image = gallery.get(sample_id)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return sample.as_dict(), self.preprocessing(buffer.getvalue(), "image/png")

