# Emotion-Analyser

A Flask-based emotion analysis web app that captures webcam frames in the browser, sends them to the backend, runs DeepFace emotion inference, and returns real-time mood-based song suggestions.

Live website: https://emotion-analyser-fb5h.onrender.com

## Highlights

- Browser-side webcam capture (cloud-safe, Render compatible)
- Frame upload API using Base64 image snapshots
- DeepFace emotion inference with robust settings
- Dynamic UI updates for detected emotion
- Stabilized song recommendations based on sustained emotion

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask, Flask-CORS
- ML/CV: DeepFace, TensorFlow, OpenCV (headless), NumPy
- Deployment: Render + Gunicorn

## Architecture

1. Browser opens webcam using getUserMedia.
2. Frontend captures a frame every 3 seconds.
3. Frame is converted to Base64 and sent to POST /process_frame.
4. Flask decodes image and runs DeepFace.analyze with enforce_detection=False.
5. Backend returns dominant emotion as JSON.
6. Frontend updates detected emotion immediately.
7. Song recommendations update only after emotion is stable for 7 seconds.

## API

### POST /process_frame

Request body:

```json
{
   "image": "data:image/jpeg;base64,..."
}
```

Success response:

```json
{
   "emotion": "happy"
}
```

Error response:

```json
{
   "error": "Missing base64 image payload."
}
```

## Local Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

Live deployment: https://emotion-analyser-fb5h.onrender.com

Important: TensorFlow does not provide wheels for Python 3.14 yet. Use Python 3.10.14.

- Build command:

```bash
bash render-build.sh
```

- Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

Render-ready files included:

- Procfile
- render-build.sh
- requirements.txt
- runtime.txt
- .python-version
- render.yaml

### Required Render Settings

- Python Version: 3.10.14
- Build Command: bash render-build.sh
- Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120

If your existing Render service still shows Python 3.14.x in logs, set Python Version manually in service settings and redeploy.

## Current Repository Structure

```text
Emotion-Analyser/
|-- app.py
|-- requirements.txt
|-- Procfile
|-- render-build.sh
|-- frontend/
|   |-- index 1.html
|   |-- index 2.html
|   |-- script.js
|   |-- styles.css
|   `-- face5.mp4
|-- legacy/
|   |-- old-ui/
|   |   |-- templates/
|   |   `-- static/
|   `-- misc/
|-- README.md
|-- Emotion Analyser Report file by Palash.pdf
|-- front page.png
`-- Result page.png
```

## Structure Notes

- The app currently prefers frontend/ for templates and static files.
- Legacy files are archived inside legacy/ to keep the root clean.
- Active runtime assets are inside frontend/ only.

## Screenshots and Report

- Front page: front page.png
- Result page: Result page.png
- Report: Emotion Analyser Report file by Palash.pdf

## Authors

- [Palash Rai](https://github.com/Palash-r26)
- Prarthana Sharma
- Sarvesh Baghel
