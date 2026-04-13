import os
import time
import base64
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR if FRONTEND_DIR.exists() else BASE_DIR / "templates"
STATIC_DIR = FRONTEND_DIR if FRONTEND_DIR.exists() else BASE_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

CORS(
    app,
    resources={r"/process_frame": {"origins": "*"}},
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

global_emotion = "Detecting..."
last_emotion = "Detecting..."
last_detection_time = time.time()
detection_interval = 5
frame_skip = 5
frame_count = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _camera_unavailable_frame(message: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera is unavailable", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    cv2.putText(frame, message, (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return b""
    return buffer.tobytes()


def generate_frames():
    global global_emotion, last_emotion, last_detection_time, frame_count

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        # Cloud runtimes (including Render) have no physical webcam.
        # In production, capture camera in the browser and POST frames to the server,
        # or run inference fully in the browser.
        offline_frame = _camera_unavailable_frame(
            "Deploy mode: stream browser/uploaded frames instead"
        )
        while True:
            if not offline_frame:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + offline_frame + b"\r\n"
            )
            time.sleep(1)
        return

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if frame_count % frame_skip == 0 and len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi_color = frame[y : y + h, x : x + w]

                try:
                    resized_face = cv2.resize(face_roi_color, (224, 224))
                    result = DeepFace.analyze(
                        resized_face,
                        actions=["emotion"],
                        enforce_detection=False,
                    )
                    emotion = result[0]["dominant_emotion"]

                    current_time = time.time()
                    if (
                        emotion != last_emotion
                        and (current_time - last_detection_time) > detection_interval
                    ):
                        last_emotion = emotion
                        last_detection_time = current_time
                        global_emotion = emotion
                except Exception as exc:
                    print("Emotion detection error:", exc)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"Emotion: {global_emotion}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        capture.release()


def _decode_base64_image(image_data: str):
    if not image_data:
        return None

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception:
        return None

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    if (TEMPLATES_DIR / "r.html").exists():
        return render_template("r.html")
    if (TEMPLATES_DIR / "index 1.html").exists():
        return render_template("index 1.html")
    return "Template not found", 404


@app.route("/analyze")
def analyze():
    if (TEMPLATES_DIR / "index 2.html").exists():
        return render_template("index 2.html")
    if (TEMPLATES_DIR / "r1.html").exists():
        return render_template("r1.html")
    return "Template not found", 404


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global global_emotion, last_emotion, last_detection_time

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not isinstance(image_data, str) or not image_data.strip():
        return jsonify({"error": "Missing base64 image payload."}), 400

    frame = _decode_base64_image(image_data)
    if frame is None:
        return jsonify({"error": "Invalid image format."}), 400

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
        )

        if isinstance(result, list):
            emotion = result[0]["dominant_emotion"]
        else:
            emotion = result["dominant_emotion"]

        global_emotion = emotion
        last_emotion = emotion
        last_detection_time = time.time()

        return jsonify({"emotion": global_emotion})
    except Exception as exc:
        return jsonify({"error": f"Emotion analysis failed: {exc}"}), 500


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/current_emotion")
def current_emotion():
    return jsonify({"emotion": global_emotion})


@app.route("/face5.mp4")
def root_face_video():
    return send_file(BASE_DIR / "face5.mp4", mimetype="video/mp4")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
