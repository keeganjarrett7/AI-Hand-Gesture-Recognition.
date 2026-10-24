
# AI Hand Gesture Recognition

A real-time computer vision and machine learning application that recognizes hand gestures using Python, OpenCV, MediaPipe, and a custom-trained Random Forest classifier.

## Overview

This project combines pretrained hand landmark detection with a supervised machine learning model trained on a custom dataset.

The application captures live webcam footage, detects 21 hand landmarks, extracts and normalizes 63 coordinate features, and classifies hand gestures in real time.

## Features

- Real-time webcam processing using OpenCV
- Hand landmark tracking using MediaPipe
- Six gesture classification categories
- Custom-trained Random Forest classifier
- Real-time gesture predictions and confidence scores
- Confidence-based rejection of uncertain predictions
- Visual landmark and finger connection overlays

## Supported Gestures

- Open Palm
- Fist
- Peace Sign
- Pointing
- Middle Finger
- Other

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- joblib

## Machine Learning Pipeline

1. OpenCV captures live webcam frames.
2. MediaPipe detects 21 hand landmarks.
3. Landmark coordinates are converted into 63 numerical features.
4. Features are normalized relative to the wrist and scaled.
5. A Random Forest classifier is trained using labeled gesture examples.
6. The trained model is saved using joblib.
7. The application loads the model and performs real-time predictions.

## Model Training and Evaluation

The initial dataset contained approximately 2,700 labeled gesture samples across six categories, collected using both hands.

A Random Forest classifier with 200 decision trees was trained using scikit-learn.

The initial model achieved approximately **97% accuracy** on a randomly held-out test set of 594 samples.

Because consecutive video frames are highly correlated, this accuracy may overestimate performance on independent recordings. Additional evaluation across different users, environments, and recording sessions is recommended.

## Installation

### 1. Clone the repository

    git clone https://github.com/keeganjarrett7/AI-Hand-Gesture-Recognition.

### 2. Create a virtual environment

    python -m venv .venv

### 3. Install dependencies

On Windows:

    .venv\Scripts\python.exe -m pip install -r requirements.txt

### 4. Download the MediaPipe model

Download the MediaPipe Hand Landmarker model from:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Place the model in the project directory as:

    hand_landmarker.task

## Running the Application

Run:

    .venv\Scripts\python.exe main.py

The application will open your webcam, detect hand landmarks, and display the predicted gesture with its confidence score.

Press **Q** to exit.

## Training a Custom Model

Run the data collection script:

    .venv\Scripts\python.exe collect_data.py

Use the following controls:

| Key | Gesture |
|-----|---------|
| 1 | Open Palm |
| 2 | Fist |
| 3 | Peace Sign |
| 4 | Pointing |
| 5 | Middle Finger |
| 6 | Other |
| 0 | Pause recording |
| Q | Quit |

Training samples are saved to `gesture_data.csv`.

Train the model:

    .venv\Scripts\python.exe train.py

The trained classifier is saved as `gesture_model.pkl`.

## Project Structure

    HandGestureRecognition/
    ├── main.py
    ├── collect_data.py
    ├── train.py
    ├── gesture_model.pkl
    ├── hand_landmarker.task
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Current Limitations

- Recognition accuracy may vary with lighting, hand orientation, and camera quality.
- Similar gestures can occasionally be misclassified.
- Unfamiliar gestures may produce incorrect predictions.
- The current implementation tracks one hand at a time.
- The initial reported accuracy is not an independent-session benchmark.

## Future Improvements

- Temporal prediction smoothing
- Simultaneous two-hand recognition
- Additional gesture categories
- More extensive model evaluation
- Neural network classification using PyTorch
- User-defined custom gestures

## Author

Keegan Jarrett

Computer Science Student  
University of South Florida

GitHub: https://github.com/keeganjarrett7
