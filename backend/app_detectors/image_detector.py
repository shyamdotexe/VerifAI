import time

import torch
from PIL import Image

from app.config import Settings
from app.exceptions import ModelUnavailableError
from .base import BaseDetector
from processing.image_processing import (
    ImageInput,
    preprocess_image,
    preprocess_pil_image,
)
from app.schemas import DetectorResult


class ImageDetector(BaseDetector):
    def __init__(self, settings: Settings | None = None, processor=None, model=None):
        settings = settings or Settings()
        super().__init__(settings.image_model_id, settings, processor, model)
        self.fake_label, self.real_label = self._labels_from_config(model.config if model is not None else None)
        if model is not None and (self.fake_label is None or self.real_label is None):
            self._unavailable = "model id2label does not safely identify fake and real classes"
            self._state = "failed"

    @staticmethod
    def _labels_from_config(config):
        return BaseDetector.resolve_binary_labels(
            config,
            {"fake", "synthetic", "deepfake", "manipulated", "aigenerated"},
            {"real", "authentic", "genuine", "human"},
        )

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.settings.model_cache_dir,
            revision=self.settings.image_model_revision,
            use_fast=False,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_id,
            cache_dir=self.settings.model_cache_dir,
            revision=self.settings.image_model_revision,
        )
        self.fake_label, self.real_label = self._labels_from_config(self.model.config)
        if self.fake_label is None or self.real_label is None:
            raise ValueError("model id2label does not safely identify fake and real classes")

    def _analyze_prepared(
        self,
        prepared: ImageInput,
        started: float,
    ) -> DetectorResult:
        self.ensure_loaded()
        if self.fake_label is None or self.real_label is None:
            raise ModelUnavailableError("Image model labels are not safely mapped")
        inputs = self.processor(images=prepared.image, return_tensors="pt")
        logits = self.predict_logits(inputs)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise ModelUnavailableError(
                "Image model did not return one binary-classification result"
            )
        probabilities = torch.softmax(logits, dim=-1)[0]
        if not bool(torch.isfinite(probabilities).all()):
            raise ModelUnavailableError("Image model returned non-finite probabilities")
        fake = float(probabilities[self.fake_label].item())
        real = float(probabilities[self.real_label].item())
        risk_score = round(fake * 100, 2)
        classification = self.classification(risk_score, self.settings)
        signal = {
            "higher_ai_risk": "The image model produced a higher synthetic-content score",
            "uncertain": "The image model produced an inconclusive synthetic-content score",
            "lower_ai_risk": "The image model produced a lower synthetic-content score",
        }[classification]
        evidence = {
            "model_id": self.model_id,
            "decoded_format": prepared.decoded_format,
            "original_mode": prepared.original_mode,
            "original_width": prepared.original_width,
            "original_height": prepared.original_height,
            "analyzed_width": prepared.analyzed_width,
            "analyzed_height": prepared.analyzed_height,
            "file_size_bytes": prepared.file_size_bytes,
            "exif_orientation_applied": prepared.exif_orientation_applied,
            "blur_variance_laplacian": prepared.blur_measurement,
            "aspect_ratio": prepared.aspect_ratio,
            "reliability_factors": prepared.reliability_factors,
        }
        return DetectorResult(
            detector="image",
            status="completed",
            classification=classification,
            risk_score=risk_score,
            reliability_score=prepared.reliability_score,
            model_output={"fake": fake, "real": real},
            signals=[signal],
            warnings=[
                "This screening result is not conclusive proof",
                "Image detector performance can vary with compression and unseen manipulation methods",
            ],
            evidence=evidence,
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def analyze_image(self, image: Image.Image) -> DetectorResult:
        started = time.perf_counter()
        prepared = preprocess_pil_image(image, self.settings)
        return self._analyze_prepared(prepared, started)

    def analyze_bytes(self, data: bytes, filename: str = "upload") -> DetectorResult:
        started = time.perf_counter()
        prepared = preprocess_image(data, filename, self.settings)
        return self._analyze_prepared(prepared, started)

    def analyze(self, path: str) -> DetectorResult:
        with Image.open(path) as image:
            return self.analyze_image(image.convert("RGB"))
