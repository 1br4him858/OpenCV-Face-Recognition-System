# Face Recognition with OpenCV (LBPH)

A small, local **face recognition programming sample** showing a three-stage pipeline: consent-based face collection, training and holdout evaluation with OpenCV's Local Binary Patterns Histograms (LBPH) recognizer, and webcam-based recognition.

## Technology
Python, OpenCV (`opencv-contrib-python`), NumPy, scikit-learn.

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
```
Use a Python version compatible with your chosen OpenCV wheel.

## Usage
1. Obtain explicit permission from each participant before collecting images.
2. Collect samples for at least two consenting people (at least five images per person):
   ```bash
   python 1_collect_faces.py --name person_01 --max-images 200
   python 1_collect_faces.py --name person_02 --max-images 200
   ```
3. Train and evaluate:
   ```bash
   python 2_train_model.py
   ```
4. Run webcam recognition:
   ```bash
   python 3_recognize_face.py --threshold 70
   ```
Press **ESC** to exit the webcam windows.

## Files
- `1_collect_faces.py`: Haar-cascade detection and grayscale face-crop collection.
- `2_train_model.py`: per-class holdout, LBPH training, classification report, and confusion matrix.
- `3_recognize_face.py`: real-time prediction with a five-frame stability check for a single face.
- `requirements.txt`: dependencies.
- `.gitignore`: excludes local face photos and model artifacts.

## Limitations and privacy
- This is a **demonstration**, not a production biometric identification or access-control system.
- The `70` LBPH distance threshold is a starting value, **not a calibrated confidence percentage**; it should be tuned on separate validation data.
- The train/test split uses different images, but images from the same capture session can be highly correlated. Reported holdout accuracy may be optimistic. Independent capture sessions are needed for credible generalization tests.
- No anti-spoofing, liveness detection, multi-person tracking, or security guarantees are provided.
- **Never upload `persons/`, real face photos, `trainer.yml`, or `names.json`** to a public repository without a lawful basis and participant permission.
- Use only on people who have explicitly consented, and comply with applicable privacy and biometric-data rules.

## Model files
`trainer.yml` and `names.json` are generated locally by the training script and are intentionally excluded from this source-code sample.
