import dlib
import cv2
import numpy as np
import os
import pickle  # Import pickle for storing embeddings
from imutils import face_utils

# Load dlib models
detector = dlib.get_frontal_face_detector()  # Face detector
shape_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")  # Landmark detector
face_rec_model = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")  # Face recognition model


def get_face_embedding(image_path):
    """Extracts 128D face embedding from an image using dlib."""
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb_image)

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None

    # Get landmarks for the first detected face
    shape = shape_predictor(rgb_image, faces[0])

    # Compute face descriptor (embedding)
    face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)

    # Convert to NumPy array
    return np.array(face_descriptor)


# Directory containing images of 10 people
image_folder = "data/Train_Images"  # Modify the path

# Dictionary to store embeddings
face_embeddings = {}

for person in os.listdir(image_folder):
    person_path = os.path.join(image_folder, person)

    if os.path.isdir(person_path):  # Ensure it's a directory
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            embedding = get_face_embedding(image_path)
            if embedding is not None:
                face_embeddings[image_name] = embedding

# Save embeddings using pickle
with open("embedding_file/face_embeddings.pkl", "wb") as f:
    pickle.dump(face_embeddings, f)

print("✅ Face embeddings saved successfully in face_embeddings.pkl")
