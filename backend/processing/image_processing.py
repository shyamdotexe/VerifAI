from dataclasses import dataclass
from io import BytesIO
import math
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.exceptions import AnalysisError


@dataclass
class ImageInput:
    image: Image.Image
    original_width: int
    original_height: int
    analyzed_width: int
    analyzed_height: int
    original_mode: str
    decoded_format: str
    file_size_bytes: int | None
    exif_orientation_applied: bool
    blur_measurement: float
    aspect_ratio: float
    reliability_score: float
    reliability_factors: dict[str, float]


def _validate_dimensions(width: int, height: int, settings: Settings) -> None:
    if width < 8 or height < 8:
        raise AnalysisError("Image dimensions are too small for meaningful analysis")
    if width > settings.max_image_width or height > settings.max_image_height:
        raise AnalysisError("Image dimensions exceed the configured limit")
    if width * height > settings.max_image_pixels:
        raise AnalysisError("Image pixel count exceeds the configured limit")


def _quality_metrics(image: Image.Image) -> tuple[float, float, float, dict[str, float]]:
    width, height = image.size
    sample = image.copy()
    sample.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    array = np.asarray(sample)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect = max(width, height) / max(1, min(width, height))

    factors = {
        "decode": 1.0,
        "resolution": min(1.0, math.sqrt((width * height) / (512 * 512))),
        "minimum_dimension": min(1.0, min(width, height) / 256),
        "aspect_ratio": max(0.0, min(1.0, (8.0 - aspect) / 6.0)),
        "sharpness": min(1.0, blur / 100.0),
    }
    reliability = 100 * (
        0.15 * factors["decode"]
        + 0.30 * factors["resolution"]
        + 0.15 * factors["minimum_dimension"]
        + 0.15 * factors["aspect_ratio"]
        + 0.25 * factors["sharpness"]
    )
    return blur, aspect, round(min(100.0, reliability), 2), factors


def _build_input(
    image: Image.Image,
    original_size: tuple[int, int],
    original_mode: str,
    decoded_format: str,
    file_size_bytes: int | None,
    orientation_applied: bool,
) -> ImageInput:
    blur, aspect, reliability, factors = _quality_metrics(image)
    return ImageInput(
        image=image,
        original_width=original_size[0],
        original_height=original_size[1],
        analyzed_width=image.width,
        analyzed_height=image.height,
        original_mode=original_mode,
        decoded_format=decoded_format,
        file_size_bytes=file_size_bytes,
        exif_orientation_applied=orientation_applied,
        blur_measurement=blur,
        aspect_ratio=aspect,
        reliability_score=reliability,
        reliability_factors=factors,
    )


def preprocess_pil_image(
    source: Image.Image,
    settings: Settings | None = None,
) -> ImageInput:
    settings = settings or Settings()
    if not isinstance(source, Image.Image):
        raise AnalysisError("Expected a Pillow image")
    try:
        original_size = source.size
        original_mode = source.mode
        _validate_dimensions(*original_size, settings)
        if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
            raise AnalysisError("Animated images are not supported")
        orientation = int(source.getexif().get(274, 1))
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        normalized.load()
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError("Image cannot be normalized") from exc

    _validate_dimensions(*normalized.size, settings)
    return _build_input(
        normalized,
        original_size,
        original_mode,
        source.format or "memory",
        None,
        orientation not in (0, 1),
    )


def preprocess_image(data: bytes, filename: str = "upload", settings: Settings | None = None) -> ImageInput:
    settings = settings or Settings()
    if not data:
        raise AnalysisError("Image file is empty")
    if len(data) > (settings.max_image_file_size_mb * 1024 * 1024):
        raise AnalysisError("Image exceeds the configured size limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as opened:
                image_format = opened.format or "unknown"
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise AnalysisError(
                        "Unsupported image format; use JPEG, PNG, or WebP"
                    )
                original_size = opened.size
                original_mode = opened.mode
                _validate_dimensions(*original_size, settings)
                opened.verify()

            with Image.open(BytesIO(data)) as opened:
                if (
                    getattr(opened, "is_animated", False)
                    or getattr(opened, "n_frames", 1) != 1
                ):
                    raise AnalysisError("Animated images are not supported")
                orientation = int(opened.getexif().get(274, 1))
                normalized = ImageOps.exif_transpose(opened).convert("RGB")
                normalized.load()
    except AnalysisError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        RuntimeError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise AnalysisError("Image is corrupted or cannot be decoded") from exc

    _validate_dimensions(*normalized.size, settings)
    return _build_input(
        normalized,
        original_size,
        original_mode,
        image_format,
        len(data),
        orientation not in (0, 1),
    )
