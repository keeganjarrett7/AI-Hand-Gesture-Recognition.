
import cv2
import mediapipe as mp
import joblib
import time
import numpy as np

from pathlib import Path

# Project file paths
PROJECT_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"
AI_MODEL_PATH = PROJECT_DIR / "gesture_model.pkl"

# Load your trained AI model
model = joblib.load(AI_MODEL_PATH)

# Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Connections between hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def extract_features(hand):
    """
    Normalize hand landmarks using the same
    method as our training data collection.
    """
    wrist = hand[0]

    coordinates = [
        (
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        )
        for point in hand
    ]

    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in coordinates
    )

    scale = max(scale, 1e-6)

    return [
        value / scale
        for point in coordinates
        for value in point
    ]


# Open webcam using Windows DirectShow
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not camera.isOpened():
    print("Could not access webcam.")
    raise SystemExit(1)

# Test the camera before running AI
success, test_frame = camera.read()

print("Camera read successful:", success)

if success:
    print("Frame shape:", test_frame.shape)
    print("Frame brightness:", test_frame.mean())
start_time = time.monotonic()

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            # Convert webcam image for MediaPipe
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            height, width, _ = frame.shape

            # Classify each detected hand
            for hand_index, hand in enumerate(
                result.hand_landmarks
            ):
                # Extract numerical features
                features = extract_features(hand)
                features_array = np.array(
                    [features], dtype=np.float32
                )

                # Ask our trained AI for probabilities
                probabilities = model.predict_proba(
                    features_array
                )[0]

                best_index = np.argmax(probabilities)
                confidence = probabilities[best_index]
                gesture = model.classes_[best_index]

                # Reject uncertain predictions
                if confidence < 0.50:
                    gesture = "Unknown"

                # Convert landmarks to pixel coordinates
                points = []

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                    cv2.circle(
                        frame, (x, y),
                        5, (0, 255, 0), -1
                    )

                # Draw hand connections
                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 255, 255),
                        2
                    )

                # Place label above the hand
                label_x = max(0, min(x for x, y in points))
                label_y = max(30, min(y for x, y in points) - 15)

                text = (
                    f"{gesture}: {confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # Display annotated video
            cv2.imshow(
                "AI Hand Gesture Recognition", frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
