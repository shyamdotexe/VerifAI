from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.config import Settings
from app.exceptions import AnalysisError
from utils.cleanup import remove_file


SUPPORTED_AUDIO_FORMATS = {"WAV", "FLAC", "OGG", "MP3", "MPEG"}


@dataclass
class AudioInput:
    waveform: np.ndarray
    original_sample_rate: int
    processed_sample_rate: int
    channels: int
    duration_seconds: float
    decoded_format: str
    silence_ratio: float
    clipping_ratio: float
    segments: list[np.ndarray]
    segment_valid_samples: list[int]
    analyzed_coverage: float
    reliability_score: float
    reliability_factors: dict[str, float]


def _validate_audio_metadata(
    sample_rate: int,
    channels: int,
    frames: int,
    settings: Settings,
) -> float:
    if sample_rate <= 0 or frames < 0:
        raise AnalysisError("Audio metadata is invalid")
    if channels <= 0 or channels > settings.max_audio_channels:
        raise AnalysisError("Audio channel count exceeds the configured limit")
    duration = frames / sample_rate
    if duration < settings.audio_min_seconds:
        raise AnalysisError("Audio is too short for meaningful analysis")
    if duration > settings.max_audio_seconds:
        raise AnalysisError("Audio exceeds the configured duration limit")
    return duration


def _prepare_waveform(
    waveform,
    sample_rate: int,
    settings: Settings,
    decoded_format: str,
) -> AudioInput:
    try:
        values = np.asarray(waveform, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Audio samples are invalid") from exc
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise AnalysisError("Audio waveform must use samples-by-channels layout")
    if values.shape[0] == 0:
        raise AnalysisError("Audio contains no samples")
    channels = int(values.shape[1])
    duration = _validate_audio_metadata(
        int(sample_rate),
        channels,
        int(values.shape[0]),
        settings,
    )
    if not bool(np.isfinite(values).all()):
        raise AnalysisError("Audio contains non-finite samples")

    clipping_ratio = float(np.mean(np.abs(values) >= 0.99))
    mono = values.mean(axis=1, dtype=np.float32)
    target = settings.audio_target_sample_rate
    if sample_rate == target:
        processed = mono.astype(np.float32, copy=False)
    else:
        divisor = math.gcd(int(sample_rate), target)
        processed = resample_poly(
            mono,
            target // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    if not bool(np.isfinite(processed).all()):
        raise AnalysisError("Audio resampling produced invalid samples")

    silence_ratio = float(np.mean(np.abs(processed) < 0.01))
    window = int(round(settings.audio_segment_seconds * target))
    minimum = int(round(settings.audio_min_seconds * target))
    segments: list[np.ndarray] = []
    valid_samples: list[int] = []
    analyzed_samples = 0
    for start in range(0, len(processed), window):
        segment = processed[start : start + window]
        if len(segment) < minimum:
            break
        analyzed_samples += len(segment)
        valid_samples.append(len(segment))
        if len(segment) < window:
            segment = np.pad(segment, (0, window - len(segment)))
        segments.append(segment.astype(np.float32, copy=False))
    if not segments:
        raise AnalysisError("Audio has no valid analysis segments")

    coverage = analyzed_samples / len(processed)
    factors = {
        "decode": 1.0,
        "usable_duration": min(1.0, duration / settings.audio_segment_seconds),
        "analyzed_coverage": coverage,
        "non_silence": 1.0 - silence_ratio,
        "not_clipped": 1.0 - clipping_ratio,
    }
    weighted = (
        0.20 * factors["decode"]
        + 0.20 * factors["usable_duration"]
        + 0.20 * factors["analyzed_coverage"]
        + 0.25 * factors["non_silence"]
        + 0.15 * factors["not_clipped"]
    )
    reliability = 100 * weighted * math.sqrt(max(0.0, factors["non_silence"]))
    return AudioInput(
        waveform=processed,
        original_sample_rate=int(sample_rate),
        processed_sample_rate=target,
        channels=channels,
        duration_seconds=duration,
        decoded_format=decoded_format,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        segments=segments,
        segment_valid_samples=valid_samples,
        analyzed_coverage=coverage,
        reliability_score=round(max(0.0, min(100.0, reliability)), 2),
        reliability_factors=factors,
    )
def preprocess_waveform(
    waveform,
    sample_rate: int,
    settings: Settings | None = None,
) -> AudioInput:
    settings = settings or Settings()
    return _prepare_waveform(waveform, sample_rate, settings, "memory")


def _decode_soundfile(
    data: bytes,
    settings: Settings,
) -> tuple[np.ndarray, int, str]:
    with sf.SoundFile(BytesIO(data)) as source:
        decoded_format = str(source.format).upper()
        if decoded_format not in SUPPORTED_AUDIO_FORMATS:
            raise AnalysisError(
                "Unsupported audio format; use WAV, FLAC, OGG, or MP3"
            )
        _validate_audio_metadata(
            int(source.samplerate),
            int(source.channels),
            int(source.frames),
            settings,
        )
        waveform = source.read(dtype="float32", always_2d=True)
        return waveform, int(source.samplerate), decoded_format


def _looks_like_mp3(data: bytes, filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return (
        suffix == ".mp3"
        or data.startswith(b"ID3")
        or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)
    )


def _decode_mp3_with_ffmpeg(
    data: bytes,
    settings: Settings,
) -> tuple[np.ndarray, int, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AnalysisError(
            "MP3 cannot be decoded by the local audio library; FFmpeg is required"
        )

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            dir=settings.temp_dir,
            delete=False,
        ) as source:
            source.write(data)
            input_path = Path(source.name)
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=settings.temp_dir,
            delete=False,
        ) as target:
            output_path = Path(target.name)

        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-t",
                str(settings.max_audio_seconds + 1),
                str(output_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise AnalysisError("MP3 cannot be decoded by FFmpeg")
        waveform, sample_rate, _ = _decode_soundfile(
            output_path.read_bytes(),
            settings,
        )
        return waveform, sample_rate, "MPEG"
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError("FFmpeg timed out while decoding MP3 audio") from exc
    finally:
        if input_path is not None:
            remove_file(input_path)
        if output_path is not None:
            remove_file(output_path)


def preprocess_audio(
    data: bytes,
    filename: str = "upload",
    settings: Settings | None = None,
) -> AudioInput:
    settings = settings or Settings()
    if not data:
        raise AnalysisError("Audio file is empty")
    if len(data) > (settings.max_audio_file_size_mb * 1024 * 1024):
        raise AnalysisError("Audio exceeds the configured size limit")
    try:
        waveform, sample_rate, decoded_format = _decode_soundfile(data, settings)
    except AnalysisError:
        raise
    except Exception as exc:
        if not _looks_like_mp3(data, filename):
            raise AnalysisError(
                "Audio is corrupted or unsupported; use WAV, FLAC, OGG, or MP3"
            ) from exc
        waveform, sample_rate, decoded_format = _decode_mp3_with_ffmpeg(
            data,
            settings,
        )
    return _prepare_waveform(
        waveform,
        sample_rate,
        settings,
        decoded_format,
    )











