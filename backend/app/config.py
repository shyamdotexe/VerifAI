from pathlib import Path


class Settings:
    image_model_id = "prithivMLmods/deepfake-detector-model-v1"
    audio_model_id = "Vansh180/deepfake-audio-wav2vec2"

    image_model_revision = None
    audio_model_revision = None

    model_cache_dir = str(Path(".model_cache"))

    risk_low_threshold = 35.0
    risk_high_threshold = 65.0

    max_audio_channels = 2
    audio_target_sample_rate = 16_000
    audio_segment_seconds = 5.0
    audio_min_seconds = 1.0
    max_audio_seconds = 120.0
    max_audio_file_size_mb = 25
    temp_dir = Path(".tmp")

    max_image_width = 4096
    max_image_height = 4096
    max_image_pixels = 16_777_216
    max_image_file_size_mb = 10