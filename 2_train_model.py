"""Train an OpenCV LBPH recognizer with a per-person holdout split."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="persons")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1:
        parser.error("--test-ratio must be between 0 and 1")
    if not hasattr(cv2, "face"):
        raise RuntimeError("Install opencv-contrib-python (not opencv-python)")
    root = Path(args.data)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {root}")
    rng = np.random.default_rng(args.seed)
    train_images, train_labels, test_images, test_labels = [], [], [], []
    names = {}
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        images = []
        for path in sorted(person.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, (150, 150)))
        if len(images) < 5:
            print(f"Skipping {person.name}: fewer than five valid images")
            continue
        label = len(names)
        names[str(label)] = person.name
        order = rng.permutation(len(images))
        n_test = max(1, min(len(images)-1, round(len(images)*args.test_ratio)))
        for i in order[n_test:]:
            train_images.append(images[i]); train_labels.append(label)
        for i in order[:n_test]:
            test_images.append(images[i]); test_labels.append(label)
    if len(names) < 2:
        raise ValueError("At least two people with five or more images each are required")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_images, np.array(train_labels, dtype=np.int32))
    predictions = [int(model.predict(image)[0]) for image in test_images]
    labels = list(range(len(names)))
    print("Holdout accuracy:", f"{accuracy_score(test_labels, predictions):.3f}")
    print(classification_report(test_labels, predictions, labels=labels,
                                target_names=[names[str(i)] for i in labels], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(test_labels, predictions, labels=labels))
    print("Note: frames from the same collection session can leak across splits; this is a demo metric, not independent-person or independent-session validation.")
    model.save("trainer.yml")
    Path("names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print("Saved trainer.yml and names.json")

if __name__ == "__main__":
    main()
