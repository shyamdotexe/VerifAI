import os
from app_detectors.image_detector import ImageDetector


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

AI_DIR = os.path.join(BASE_DIR, "test_media", "images", "ai")
REAL_DIR = os.path.join(BASE_DIR, "test_media", "images", "real")


test_files = []

for filename in os.listdir(AI_DIR):
    test_files.append(
        (os.path.join(AI_DIR, filename), "ai")
    )

for filename in os.listdir(REAL_DIR):
    test_files.append(
        (os.path.join(REAL_DIR, filename), "real")
    )


detector = ImageDetector()

passed = 0
total = len(test_files)


for path, expected in test_files:

    print("=" * 60)
    print(f"File: {os.path.basename(path)}")
    print(f"Expected: {expected}")

    try:
        result = detector.analyze(path)

        predicted = result.classification

        print(f"Predicted: {predicted}")
        print(f"Risk Score: {result.risk_score}%")
        print(f"Reliability: {result.reliability_score}")

        if expected == "ai":
            correct = predicted == "higher_ai_risk"
        else:
            correct = predicted == "lower_ai_risk"

        if correct:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")

    except Exception as exc:
        print(f"ERROR: {exc}")


print()
print("=" * 60)
print("IMAGE DETECTOR TEST SUMMARY")
print("=" * 60)
print(f"Passed: {passed}/{total}")

if total:
    print(f"Accuracy: {(passed / total) * 100:.2f}%")
else:
    print("Accuracy: 0.00%")

print("=" * 60)