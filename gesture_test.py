import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import HandFeatures
from gesture_recognizer import GestureRecognizer

from interaction import (
    InteractionEngine,
    VirtualObject
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,

    # Track one hand
    num_hands=1,

    # Detection confidence
    min_hand_detection_confidence=0.5,

    # Hand presence confidence
    min_hand_presence_confidence=0.5,

    # Tracking confidence
    min_tracking_confidence=0.5
)


detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# AETHER COMPONENTS
# ============================================================

# Converts landmarks → useful measurements
features = HandFeatures()


# Converts measurements → gestures
recognizer = GestureRecognizer()


# Handles virtual-world interaction
interaction = InteractionEngine()


# ============================================================
# CREATE VIRTUAL OBJECT
# ============================================================

object1 = VirtualObject(
    x=640,
    y=360,
    radius=60
)

interaction.add_object(
    object1
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Could not open camera.")

    exit()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    success, frame = camera.read()


    if not success:

        print("Could not read from camera.")

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert to MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # HAND DETECTION
    # ========================================================

    result = detector.detect(
        mp_image
    )


    # ========================================================
    # HAND FOUND
    # ========================================================

    if result.hand_landmarks:

        # Get first detected hand
        hand = result.hand_landmarks[0]


        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        hand_features = features.update(
            hand
        )


        # ----------------------------------------------------
        # Get normalized position
        # ----------------------------------------------------

        normalized_x = hand_features["x"]
        normalized_y = hand_features["y"]


        # ----------------------------------------------------
        # Convert normalized → screen coordinates
        # ----------------------------------------------------

        screen_width = frame.shape[1]
        screen_height = frame.shape[0]


        screen_x = int(
            normalized_x * screen_width
        )

        screen_y = int(
            normalized_y * screen_height
        )


        # ====================================================
        # GESTURE RECOGNITION
        # ====================================================

        gesture = recognizer.recognize(
            hand_features
        )


        # ====================================================
        # PROCESS GESTURE EVENT
        # ====================================================

        if gesture:

            print(
                "GESTURE:",
                gesture
            )


            interaction.process_event(
                gesture,
                screen_x,
                screen_y
            )


        # ====================================================
        # UPDATE INTERACTION
        # ====================================================

        interaction.update(
            screen_x,
            screen_y,
            recognizer.is_pinching
        )


        # ====================================================
        # DRAW INDEX FINGER CURSOR
        # ====================================================

        cv2.circle(
            frame,

            (
                screen_x,
                screen_y
            ),

            12,

            (0, 255, 0),

            -1
        )


        # ====================================================
        # DRAW PINCH STATUS
        # ====================================================

        if recognizer.is_pinching:

            cv2.putText(
                frame,

                "PINCHING",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0, 255, 0),

                2
            )

        else:

            cv2.putText(
                frame,

                "OPEN",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (255, 255, 255),

                2
            )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        features.reset()


    # ========================================================
    # DRAW VIRTUAL OBJECT
    # ========================================================

    cv2.circle(
        frame,

        (
            int(object1.x),
            int(object1.y)
        ),

        object1.radius,

        (255, 0, 0),

        -1
    )


    # ========================================================
    # DRAW OBJECT LABEL
    # ========================================================

    cv2.putText(
        frame,

        "AETHER OBJECT",

        (
            int(object1.x) - 80,
            int(object1.y) - 75
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Aether",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()