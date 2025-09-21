import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
import threading

# Fix OpenMP error for some torch/omp combos
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ANTI_SPOOF_DIR = os.path.join(ROOT_DIR, "Silent-Face-Anti-Spoofing")
RESOURCES_DIR = os.path.join(ANTI_SPOOF_DIR, "resources")
ANTI_SPOOF_MODELS_DIR = os.path.join(RESOURCES_DIR, "anti_spoof_models")
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "Train_Images")

# Add Anti-Spoofing repo to path and import test and Detection
sys.path.append(ANTI_SPOOF_DIR)
from test import test  # noqa: E402
from src.anti_spoof_predict import Detection  # noqa: E402


def ensure_cv2_face():
    if not hasattr(cv2, 'face'):
        st.error("OpenCV 'face' module not found. Please install opencv-contrib-python.")
        st.stop()


def load_and_train_recognizer():
    """Train an LBPH face recognizer from TRAIN_IMAGES_DIR using Detection bbox."""
    ensure_cv2_face()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = []
    labels = []
    label_names = []
    label_map = {}
    detector = Detection()

    if not os.path.isdir(TRAIN_IMAGES_DIR):
        st.warning("Training images folder not found. Face recognition will default to Unknown.")
        return recognizer, label_names

    current_label = 0
    for person in sorted(os.listdir(TRAIN_IMAGES_DIR)):
        person_path = os.path.join(TRAIN_IMAGES_DIR, person)
        if not os.path.isdir(person_path):
            continue
        label_map[person] = current_label
        label_names.append(person)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
            try:
                x, y, w, h = detector.get_bbox(img)
                face = img[max(0,y):y+h, max(0,x):x+w]
            except Exception:
                # Fallback: use full image
                face = img
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            images.append(gray)
            labels.append(current_label)
        current_label += 1

    if images and labels:
        recognizer.train(images, np.array(labels))
    else:
        st.warning("No training images found; recognition may always be Unknown.")
    return recognizer, label_names


def recognize_face_lbph(frame, recognizer, label_names, detector: Detection, conf_threshold: float = 60.0):
    try:
        x, y, w, h = detector.get_bbox(frame)
        face = frame[max(0,y):y+h, max(0,x):x+w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))
        if len(label_names) == 0:
            return "Unknown", (0, 0, 255)
        label_id, confidence = recognizer.predict(gray)
        if confidence <= conf_threshold and 0 <= label_id < len(label_names):
            return label_names[label_id], (0, 255, 0)
    except Exception:
        pass
    return "Unknown", (0, 0, 255)


# Streamlit UI
st.title("Real-Time Face Recognition")
st.write("Turn on your webcam and authenticate yourself.")

# Train recognizer from dataset
recognizer, person_names = load_and_train_recognizer()
detector_for_bbox = Detection()

run = st.button("Start Webcam", use_container_width=True)
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce lag
    cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust FPS for smoother video
    frame_placeholder = st.empty()
    authenticated_user = None
    last_processed_frame_time = time.time()
    color = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Error accessing webcam.")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for better performance

        # Process only every ~0.1s to optimize performance
        if time.time() - last_processed_frame_time > 0.1:
            last_processed_frame_time = time.time()
            # Anti-spoofing check first
            spoof_label = test(image=frame, model_dir=ANTI_SPOOF_MODELS_DIR, device_id=0)
            if spoof_label == 1:
                best_match, color = recognize_face_lbph(frame, recognizer, person_names, detector_for_bbox)
                if best_match not in ("Unknown", "Spoofer"):
                    authenticated_user = best_match
                    st.success(f"✅ User Authenticated: {authenticated_user}")
                    frame_placeholder.empty()
                    time.sleep(2)
                    break
            else:
                best_match, color = ("Spoofer", (0, 0, 255))
            

        # Draw bounding box
        try:
            x, y, w, h = detector_for_bbox.get_bbox(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Spoofer" if color == (0, 0, 255) else "Genuine",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            pass

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    if authenticated_user:
        st.success(f"🔹 Welcome, {authenticated_user}! Access granted.")
    else:
        st.error("❌ Access Denied: Face not recognized.")
