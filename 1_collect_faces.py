"""Collect consented face crops for a local LBPH demo."""
import argparse
from pathlib import Path
import re
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Consenting participant identifier")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", args.name):
        parser.error("Name must contain only letters, digits, _ or -")
    directory = Path("persons") / args.name
    directory.mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade")
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")
    count = len(list(directory.glob("*.jpg")))
    try:
        while count < args.max_images:
            ok, frame = camera.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
                target = directory / f"face_{count:05d}.jpg"
                if cv2.imwrite(str(target), face):
                    count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Saved: {count}/{args.max_images} | ESC to stop", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Face collection - consent required", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} total images in {directory}")

if __name__ == "__main__":
    main()
