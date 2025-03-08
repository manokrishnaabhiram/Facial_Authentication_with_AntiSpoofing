# Facial_Recognition_Authentication
This project implements a real-time face recognition system with anti-spoofing detection using Dlib, OpenCV, FAISS, and Streamlit. It extracts face embeddings from images, stores them in a database, and then performs real-time recognition using a webcam while preventing spoofing attacks.

# README
---

## Overview
This project consists of two Python scripts for face recognition and anti-spoofing using Dlib and Streamlit.

### **1. Face Embedding Extraction (1st Code)**
This script extracts 128-dimensional face embeddings from images and saves them as a pickle file for later use in face recognition.

#### **Dependencies**
- OpenCV (`cv2`)
- Dlib
- NumPy
- Imutils
- Pickle
- OS

#### **How It Works**
1. Loads pre-trained Dlib models:
   - `shape_predictor_68_face_landmarks.dat`: Landmark detector
   - `dlib_face_recognition_resnet_model_v1.dat`: Face recognition model
2. Iterates through images in `data/Train_Images`.
3. Detects faces and extracts 128D embeddings.
4. Saves embeddings in `embedding_file/face_embeddings.pkl` using `pickle`.

#### **Output**
- A pickle file (`face_embeddings.pkl`) containing extracted embeddings for all images.

---

### **2. Real-Time Face Recognition with Anti-Spoofing (2nd Code)**
This script loads the stored embeddings and performs real-time face recognition while checking for spoofing attacks using an anti-spoofing model.

#### **Dependencies**
- Streamlit (`st`)
- OpenCV (`cv2`)
- Dlib
- NumPy
- Pickle
- FAISS (for fast similarity search)
- OS, Sys, Time, Threading

#### **How It Works**
1. Loads stored embeddings from `face_embeddings.pkl`.
2. Converts embeddings into a FAISS index for fast similarity search.
3. Captures real-time webcam frames.
4. Runs anti-spoofing detection using `Silent-Face-Anti-Spoofing`.
5. If genuine:
   - Extracts the face embedding.
   - Searches for the closest match in stored embeddings.
   - If similarity is above the threshold (0.45), the user is authenticated.
6. Displays real-time results with bounding boxes:
   - **Green**: Genuine user
   - **Red**: Spoofer or unknown user

#### **Output**
- Streamlit web app displaying real-time face recognition and anti-spoofing results.
- User authentication messages.

---

### **Setup & Execution**

#### **1. Install Dependencies**
```bash
pip install opencv-python dlib numpy imutils pickle-mixin faiss-cpu streamlit
```

#### **2. Run Face Embedding Extraction**
```bash
python face_embedding_extraction.py
```

#### **3. Run Real-Time Face Recognition**
```bash
streamlit run face_recognition.py
```

---

### **Project Structure**
```
project-folder/
│-- model/
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│-- data/
│   ├── Train_Images/
│-- embedding_file/
│   ├── face_embeddings.pkl
│-- face_embedding_extraction.py
│-- face_recognition.py
│-- Silent-Face-Anti-Spoofing/
```

---

### **Notes**
- Ensure `model/` contains the required Dlib models.
- Store training images in `data/Train_Images/` (one folder per person).
- The anti-spoofing model should be placed inside `Silent-Face-Anti-Spoofing/`.

**Author:** [Your Name]

