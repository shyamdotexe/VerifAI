from pathlib import Path

from app_detectors.audio_detector import AudioDetector

DATASET = Path("test_media/audio")

detector = AudioDetector()

passed = 0
total = 0

for expected in ["ai", "real"]:
    folder = DATASET / expected

    if not folder.exists():
        continue

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue

        total += 1

        try:
            result = detector.analyze(str(file))

            prediction = result.classification
            score = result.risk_score
            confidence = result.reliability_score

            # Map detector classification to expected dataset labels
            if prediction == "higher_ai_risk":
                predicted_label = "ai"
            elif prediction == "lower_ai_risk":
                predicted_label = "real"
            else:
                predicted_label = "uncertain"

            passed_test = predicted_label == expected

            if passed_test:
                passed += 1

            print("=" * 60)
            print(f"File: {file.name}")
            print(f"Expected: {expected}")
            print(f"Prediction: {predicted_label}")
            print(f"Classification: {prediction}")
            print(f"Risk Score: {score}%")
            print(f"Reliability: {confidence}")
            print(f"RESULT: {'PASS' if passed_test else 'FAIL'}")

        except Exception as e:
            print("=" * 60)
            print(f"File: {file.name}")
            print(f"ERROR: {e}")

print()
print("=" * 60)
print("AUDIO DETECTOR TEST SUMMARY")
print("=" * 60)
print(f"Passed: {passed}/{total}")

if total:
    print(f"Accuracy: {passed / total * 100:.2f}%")
else:
    print("Accuracy: 0.00%")

print("=" * 60)