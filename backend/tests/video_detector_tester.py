from app_detectors.video_detector import VideoDetector


VIDEO_PATH = "C:/backend/uploads/fake_03.mp4"


detector = VideoDetector()

result = detector.analyze(
    VIDEO_PATH
)

print()
print(
    "=============================="
)
print(
    "FINAL VIDEO RESULT"
)
print(
    "=============================="
)

print(
    "Modality:",
    result["modality"]
)

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Suspicion Score:",
    f"{result['score']}%"
)

print(
    "Confidence:",
    f"{result['confidence']}%"
)

print(
    "Risk:",
    result["risk_level"]
)

print(
    "Clips:",
    result["clips_analyzed"]
)

print()
print(
    "Suspicious Clips:"
)

for clip in result[
    "suspicious_clips"
]:

    print(
        f"Clip {clip['clip_index']} | "
        f"{clip['timestamp']:.2f}s | "
        f"Fake: "
        f"{clip['fake_probability'] * 100:.2f}%"
    )