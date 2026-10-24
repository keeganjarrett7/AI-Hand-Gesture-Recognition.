
import cv2
import csv
import time
import mediapipe as mp
from pathlib import Path

# Project file locations
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"
DATA_PATH = PROJECT_DIR / "gesture_data.csv"

# Keyboard controls for different gestures
GESTURES = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "peace",
    ord("4"): "pointing",
    ord("5"): "middle_finger",
    ord("6"): "other"
}

# Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


def extract_features(hand):
    """
    Convert 21 landmarks into 63 normalized features.
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

    # Normalize coordinates by hand size
    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in coordinates
    )

    scale = max(scale, 1e-6)

    features = [
        value / scale
        for point in coordinates
        for value in point
    ]

    return features


# Create CSV if it doesn't already exist
if not DATA_PATH.exists():
    with open(DATA_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["label"] + [f"feature_{i}" for i in range(63)]
        )


# Open webcam (Windows DirectShow)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit(1)

start_time = time.monotonic()
last_save = 0
sample_count = 0
selected_gesture = None

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            # Display current recording status
            label = selected_gesture or "Paused"

            cv2.putText(
                frame,
                f"Gesture: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples collected: {sample_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Training Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                break

            # Pause recording
            if key == ord("0"):
                selected_gesture = None

            # Select gesture
            elif key in GESTURES:
                selected_gesture = GESTURES[key]
                last_save = time.monotonic()

            # Record normalized hand landmarks
            if (
                selected_gesture is not None
                and result.hand_landmarks
                and time.monotonic() - last_save >= 0.15
            ):
                hand = result.hand_landmarks[0]
                features = extract_features(hand)

                with open(DATA_PATH, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [selected_gesture] + features
                    )

                sample_count += 1
                last_save = time.monotonic()

finally:
    camera.release()
    cv2.destroyAllWindows()
