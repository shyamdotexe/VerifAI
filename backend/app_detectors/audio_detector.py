import time
from pathlib import Path

import numpy as np
import torch

from app.config import Settings
from app.exceptions import AnalysisError, ModelUnavailableError
from .base import BaseDetector
from processing.audio_processing import (
    AudioInput,
    preprocess_audio,
    preprocess_waveform,
)
from app.schemas import DetectorResult


class AudioDetector(BaseDetector):
    def __init__(self, settings: Settings | None = None, processor=None, model=None):
        settings = settings or Settings()
        super().__init__(settings.audio_model_id, settings, processor, model)
        self.spoof_label, self.bonafide_label = self._labels_from_config(
            model.config if model is not None else None
        )
        if model is not None and (self.spoof_label is None or self.bonafide_label is None):
            self._unavailable = "model id2label does not safely identify spoof and bonafide classes"
            self._state = "failed"

    @staticmethod
    def _labels_from_config(config):
        return BaseDetector.resolve_binary_labels(
            config,
            {"spoof", "fake", "synthetic", "deepfake", "manipulated"},
            {"bonafide", "real", "authentic", "genuine", "human"},
        )

    def _load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                cache_dir=self.settings.model_cache_dir,
                revision=self.settings.audio_model_revision,
            )
        except Exception:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.settings.model_cache_dir,
                revision=self.settings.audio_model_revision,
            )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id,
            cache_dir=self.settings.model_cache_dir,
            revision=self.settings.audio_model_revision,
        )
        self.spoof_label, self.bonafide_label = self._labels_from_config(self.model.config)
        if self.spoof_label is None or self.bonafide_label is None:
            raise ValueError("model id2label does not safely identify spoof and bonafide classes")

    def _score_segments(
        self,
        segments: list[np.ndarray],
        sample_rate: int,
    ) -> tuple[list[float], list[float]]:
        self.ensure_loaded()
        if self.spoof_label is None or self.bonafide_label is None:
            raise ModelUnavailableError("Audio model labels are not safely mapped")
        spoof_scores: list[float] = []
        bonafide_scores: list[float] = []
        for segment in segments:
            inputs = self.processor(segment, sampling_rate=sample_rate, return_tensors="pt")
            logits = self.predict_logits(inputs)
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
                raise ModelUnavailableError(
                    "Audio model did not return one binary-classification result"
                )
            probabilities = torch.softmax(logits, dim=-1)[0]
            if not bool(torch.isfinite(probabilities).all()):
                raise ModelUnavailableError(
                    "Audio model returned non-finite probabilities"
                )
            spoof_scores.append(float(probabilities[self.spoof_label].item()))
            bonafide_scores.append(
                float(probabilities[self.bonafide_label].item())
            )
        return spoof_scores, bonafide_scores

    def _result(
        self,
        prepared: AudioInput,
        spoof_scores: list[float],
        bonafide_scores: list[float],
        started: float,
    ) -> DetectorResult:
        spoof = float(np.median(spoof_scores))
        bonafide = float(np.median(bonafide_scores))
        risk_score = round(spoof * 100, 2)
        classification = self.classification(risk_score, self.settings)
        signal = {
            "higher_ai_risk": "The audio model produced a higher spoof score",
            "uncertain": "The audio model produced an inconclusive spoof score",
            "lower_ai_risk": "The audio model produced a lower spoof score",
        }[classification]
        warnings = [
            "This screening result is not conclusive proof",
            (
                "This detector is intended for speech and may be less reliable for "
                "music, environmental audio, heavy compression, strong background "
                "noise, and unseen voice-generation methods"
            ),
        ]
        if prepared.duration_seconds < self.settings.audio_segment_seconds:
            warnings.append("Very short recordings provide less analysis coverage")
        if prepared.silence_ratio > 0.5:
            warnings.append("A large portion of the recording is silent or near-silent")
        if prepared.clipping_ratio > 0.01:
            warnings.append("The recording contains substantial clipping")

        threshold = self.settings.risk_high_threshold / 100
        evidence = {
            "model_id": self.model_id,
            "decoded_format": prepared.decoded_format,
            "original_sample_rate": prepared.original_sample_rate,
            "processed_sample_rate": prepared.processed_sample_rate,
            "channels": prepared.channels,
            "duration_seconds": prepared.duration_seconds,
            "silence_ratio": prepared.silence_ratio,
            "clipping_ratio": prepared.clipping_ratio,
            "analyzed_coverage": prepared.analyzed_coverage,
            "segment_count": len(spoof_scores),
            "segment_valid_samples": prepared.segment_valid_samples,
            "segment_spoof_scores": spoof_scores,
            "segment_bonafide_scores": bonafide_scores,
            "peak_spoof_probability": max(spoof_scores),
            "segments_above_high_threshold": sum(
                score > threshold for score in spoof_scores
            ),
            "aggregation": "median spoof probability",
            "reliability_factors": prepared.reliability_factors,
        }
        return DetectorResult(
            detector="audio",
            status="completed",
            classification=classification,
            risk_score=risk_score,
            reliability_score=prepared.reliability_score,
            model_output={"spoof": spoof, "bonafide": bonafide},
            signals=[signal],
            warnings=warnings,
            evidence=evidence,
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _analyze_prepared(
        self,
        prepared: AudioInput,
        started: float,
    ) -> DetectorResult:
        spoof_scores, bonafide_scores = self._score_segments(
            prepared.segments,
            prepared.processed_sample_rate,
        )
        return self._result(
            prepared,
            spoof_scores,
            bonafide_scores,
            started,
        )

    def analyze_waveform(self, waveform, sample_rate: int) -> DetectorResult:
        started = time.perf_counter()
        prepared = preprocess_waveform(waveform, sample_rate, self.settings)
        return self._analyze_prepared(prepared, started)

    def analyze_bytes(self, data: bytes, filename: str = "upload") -> DetectorResult:
        started = time.perf_counter()
        prepared = preprocess_audio(data, filename, self.settings)
        return self._analyze_prepared(prepared, started)

    def analyze_path(self, path: str) -> DetectorResult:
        source_path = Path(path)
        max_bytes = self.settings.max_audio_file_size_mb * 1024 * 1024
        try:
            size = source_path.stat().st_size
        except OSError as exc:
            raise AnalysisError("Audio path cannot be read") from exc
        if size > max_bytes:
            raise AnalysisError("Audio exceeds the configured size limit")
        try:
            with source_path.open("rb") as source:
                data = source.read(max_bytes + 1)
        except OSError as exc:
            raise AnalysisError("Audio path cannot be read") from exc
        return self.analyze_bytes(data, source_path.name)

    def analyze(self, path: str) -> DetectorResult:
        return self.analyze_path(path)
