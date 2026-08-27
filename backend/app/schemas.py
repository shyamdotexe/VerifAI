from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectorResult:
    detector: str = ""
    status: str = ""
    classification: str = ""
    risk_score: float = 0.0
    reliability_score: float = 0.0
    model_output: dict[str, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    processing_ms: float = 0.0