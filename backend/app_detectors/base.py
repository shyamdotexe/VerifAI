from abc import ABC, abstractmethod
import re
from threading import Lock

import torch

from app.config import Settings
from app.exceptions import ModelUnavailableError


class BaseDetector(ABC):
    def __init__(self, model_id: str, settings: Settings | None = None, processor=None, model=None):
        self.model_id = model_id
        self.settings = settings or Settings()
        self.processor = processor
        self.model = model
        self._lock = Lock()
        self._inference_lock = Lock()
        self._unavailable: str | None = None
        self._state = "not_loaded"
        self._initialized = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def classification(risk: float, settings: Settings) -> str:
        if risk < settings.risk_low_threshold:
            return "lower_ai_risk"
        if risk <= settings.risk_high_threshold:
            return "uncertain"
        return "higher_ai_risk"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def status(self) -> dict[str, str | bool | None]:
        available = None
        if self._state == "ready":
            available = True
        elif self._state == "failed":
            available = False
        return {
            "model_id": self.model_id,
            "state": self._state,
            "loaded": self._state == "ready",
            "available": available,
            "device": str(self._device),
            "error": "Model initialization failed" if self._state == "failed" else "",
        }

    @staticmethod
    def resolve_binary_labels(
        config,
        risk_aliases: set[str],
        authentic_aliases: set[str],
    ) -> tuple[int | None, int | None]:
        mapping = getattr(config, "id2label", {}) or {}
        if len(mapping) != 2:
            return None, None

        risk_matches: list[int] = []
        authentic_matches: list[int] = []
        for raw_index, raw_label in mapping.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                return None, None
            normalized = re.sub(r"[^a-z0-9]+", "", str(raw_label).lower())
            if normalized in risk_aliases:
                risk_matches.append(index)
            if normalized in authentic_aliases:
                authentic_matches.append(index)

        if len(risk_matches) != 1 or len(authentic_matches) != 1:
            return None, None
        if risk_matches[0] == authentic_matches[0]:
            return None, None
        return risk_matches[0], authentic_matches[0]

    @abstractmethod
    def _load(self) -> None:
        raise NotImplementedError

    def ensure_loaded(self, retry: bool = False) -> None:
        if self._state == "failed" and not retry:
            raise ModelUnavailableError(self._unavailable)
        if self._initialized and self.is_loaded:
            return
        with self._lock:
            if self._initialized and self.is_loaded:
                return
            if self._state == "failed" and not retry:
                raise ModelUnavailableError(self._unavailable)
            if retry:
                self._unavailable = None
            self._state = "loading"
            try:
                if not self.is_loaded:
                    self._load()
                self.model.eval()
                self.model.to(self._device)
                self._initialized = True
                self._state = "ready"
            except Exception as exc:
                self._unavailable = f"Model '{self.model_id}' is unavailable: {exc}"
                self._state = "failed"
                raise ModelUnavailableError(self._unavailable) from exc

    def predict_logits(self, inputs):
        self.ensure_loaded()
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._inference_lock, torch.inference_mode():
            return self.model(**inputs).logits
