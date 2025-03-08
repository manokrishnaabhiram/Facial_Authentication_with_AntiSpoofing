import streamlit as st
import dlib
import cv2
import numpy as np
import pickle
import faiss
import sys
import os
import time
import threading

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Silent-Face-Anti-Spoofing"))
from test import test

# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

# Load stored embeddings
with open("embedding_file/face_embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

# Convert embeddings to FAISS index
person_names = [name.rsplit('.', 1)[0] for name in stored_embeddings.keys()]
embedding_matrix = np.array(list(stored_embeddings.values()), dtype=np.float32)

d = 128  # Embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embedding_matrix)

# Strict match threshold
STRICT_MATCH_THRESHOLD = 0.45


def get_face_embedding(frame):
    """Extracts a 128D face embedding from a webcam frame using dlib."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    if len(faces) == 0:
        return None, None

    shape = shape_predictor(rgb_frame, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(rgb_frame, shape)
    return np.array(face_descriptor, dtype=np.float32), faces[0]


def process_frame(frame):
    """Process frame in a separate thread to improve performance."""
    label = test(
        image=frame,
        model_dir='C:\\Users\\Nikhil\\PycharmProjects\\Ideathon\\Facial_rec5\\Silent-Face-Anti-Spoofing\\resources\\anti_spoof_models',
        device_id=0
    )

    if label == 1:
        embedding, face_rect = get_face_embedding(frame)

        if embedding is not None:
            embedding = embedding.reshape(1, -1)
            distances, indices = index.search(embedding, 1)

            if distances[0][0] < STRICT_MATCH_THRESHOLD:
                return person_names[indices[0][0]], (0, 255, 0)  # Green for genuine users
            else:
                return "Unknown", (0, 0, 255)  # Red for unknown users
    return "Spoofer", (0, 0, 255)  # Red for spoofers


# Streamlit UI
st.title("Real-Time Face Recognition")
st.write("Turn on your webcam and authenticate yourself.")

run = st.button("Start Webcam", use_container_width=True)
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce lag
    cap.set(cv2.CAP_PROP_FPS, 30)  # Adjust FPS for smoother video
    frame_placeholder = st.empty()
    authenticated_user = None
    last_processed_frame_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Error accessing webcam.")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize for better performance

        # Process only every 3rd frame to optimize performance
        if time.time() - last_processed_frame_time > 0.1:
            last_processed_frame_time = time.time()
            result = process_frame(frame)

            if result:
                best_match, color = result

                if best_match != "Unknown" and best_match != "Spoofer":
                    authenticated_user = best_match
                    st.success(f"✅ User Authenticated: {authenticated_user}")
                    frame_placeholder.empty()  # Clear the last frame
                    time.sleep(2)  # Short delay before closing camera
                    break
            else:
                color = (0, 0, 255)

        # Draw bounding box
        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Spoofer" if color == (0, 0, 255) else "Genuine",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()

    if authenticated_user:
        st.success(f"🔹 Welcome, {authenticated_user}! Access granted.")
    else:
        st.error("❌ Access Denied: Face not recognized.")
