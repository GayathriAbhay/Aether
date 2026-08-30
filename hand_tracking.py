import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gesture_recognizer import GestureRecognizer


MODEL_PATH = "models/hand_landmarker.task"


# Create the model configuration
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# Create the detector
detector = vision.HandLandmarker.create_from_options(
    options
)


# Open webcam
camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read from camera.")
        break


    # OpenCV → BGR
    # MediaPipe → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert frame into a MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hand
    result = detector.detect(mp_image)


    # Check if a hand was detected
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Index fingertip
        index_tip = hand[8]

        print(
            "Index:",
            index_tip.x,
            index_tip.y,
            index_tip.z
        )


    cv2.imshow("Aether", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
recognizer = GestureRecognizer()
