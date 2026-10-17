"""Live face recognition demo using trained LBPH weights."""
import argparse
import json
from collections import deque
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="LBPH distance cutoff (lower is a better match); tune on held-out data")
    args = parser.parse_args()
    if not hasattr(cv2, "face"):
        raise RuntimeError("Install opencv-contrib-python")
    if not Path("trainer.yml").exists() or not Path("names.json").exists():
        raise FileNotFoundError("Run 2_train_model.py first")
    names = json.loads(Path("names.json").read_text(encoding="utf-8"))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade")
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")
    history = deque(maxlen=5)
    last_announced = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            # Require exactly one face for a meaningful single-person confirmation.
            current_name = None
            for x, y, w, h in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
                label, distance = recognizer.predict(roi)
                name = names.get(str(label), "Unknown") if distance < args.threshold else "Unknown"
                if len(faces) == 1:
                    current_name = name
                color = (0,255,0) if name != "Unknown" else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{name} ({distance:.1f})", (x, max(20,y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            history.append(current_name)
            if len(history) == 5 and len(set(history)) == 1 and history[0] is not None:
                if history[0] != last_announced:
                    print("Stable prediction:", history[0])
                    last_announced = history[0]
            else:
                last_announced = None
            cv2.imshow("Face Recognition Demo | ESC to quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
