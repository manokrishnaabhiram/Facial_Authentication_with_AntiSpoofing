# Facial_Authentication_with_AntiSpoofing
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
- CMake (Required for `facial_recognition` library)

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
pip install opencv-python dlib numpy imutils pickle-mixin faiss-cpu streamlit cmake
```

📌 **Watch these video tutorials for required installations:**
- **Visual Studio (Required for Dlib setup):** [YouTube Installation Guide](https://youtu.be/oTv7HB6CRpQ?si=8mRwcIC6KwU3SrR3)
- **CMake Installation Guide:** [YouTube CMake Setup](https://youtu.be/8_X5Iq9niDE?si=Tytq4FnzvnA_WDO7)

#### **2. Download Required Models**
- **Face Recognition Model:** [dlib_face_recognition_resnet_model_v1.dat](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat)
- **Landmark Detector Model:** [shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2/)

Extract the `shape_predictor_68_face_landmarks.dat.bz2` file and place both models inside the `model/` directory.

#### **3. Create Required Folders**
Before executing the scripts, ensure that the following folders exist:
```bash
mkdir model embedding_file
```

#### **4. Run Face Embedding Extraction**
```bash
python create_image_embeddings.py
```

#### **5. Run Real-Time Face Recognition**
```bash
streamlit run main.py
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
- CMake and Visual Studio are required for the `facial_recognition` library.
- **Ensure `model/` and `embedding_file/` directories exist before execution.**

**Author:** [Nikhil Anumalla]
