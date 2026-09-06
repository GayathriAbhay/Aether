import cv2
import mediapipe as mp
import numpy as np
import json

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze import GazeEstimator
from calibration import GazeCalibration

from features import HandFeatures
from gesture_recognizer import GestureRecognizer

from interaction import (
    InteractionEngine,
    VirtualObject
)

from intent import IntentEngine


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


FACE_MODEL = (
    BASE_DIR
    / "models"
    / "face_landmarker.task"
)


HAND_MODEL = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


CALIBRATION_FILE = (
    BASE_DIR
    / "gaze_calibration.json"
)


# ============================================================
# SCREEN
# ============================================================

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


# ============================================================
# LOAD CALIBRATION
# ============================================================

with open(
    CALIBRATION_FILE,
    "r"
) as file:

    calibration_data = json.load(file)


calibration = GazeCalibration()


calibration.matrix = np.array(
    calibration_data["matrix"],
    dtype=np.float64
)


# ============================================================
# FACE LANDMARKER
# ============================================================

face_base_options = python.BaseOptions(
    model_asset_path=str(
        FACE_MODEL
    )
)


face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


face_detector = (
    vision.FaceLandmarker
    .create_from_options(
        face_options
    )
)


# ============================================================
# HAND LANDMARKER
# ============================================================

hand_base_options = python.BaseOptions(
    model_asset_path=str(
        HAND_MODEL
    )
)


hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


hand_detector = (
    vision.HandLandmarker
    .create_from_options(
        hand_options
    )
)


# ============================================================
# SYSTEMS
# ============================================================

gaze_estimator = GazeEstimator()

hand_features = HandFeatures()

gesture_recognizer = GestureRecognizer()

intent_engine = IntentEngine()

interaction = InteractionEngine()


# ============================================================
# OBJECTS
# ============================================================

object1 = VirtualObject(
    x=350,
    y=300,
    radius=70,
    name="Aether Core"
)


object2 = VirtualObject(
    x=800,
    y=250,
    radius=60,
    name="Energy Orb"
)


object3 = VirtualObject(
    x=600,
    y=520,
    radius=80,
    name="Portal"
)


interaction.add_object(
    object1
)

interaction.add_object(
    object2
)

interaction.add_object(
    object3
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "Could not open camera."
    )

    exit()


print()
print("================================")
print("          AETHER")
print("================================")
print()
print("Gaze + Gesture interaction")
print()
print("Look at an object.")
print("Pinch to grab.")
print("Move your hand to move it.")
print("Release pinch to drop it.")
print()
print("Press Q to quit.")
print()


# ============================================================
# CURSOR
# ============================================================

cursor_x = SCREEN_WIDTH // 2

cursor_y = SCREEN_HEIGHT // 2


cursor_alpha = 0.18


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "Could not read camera."
        )

        break


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # FACE
    # ========================================================

    face_result = (
        face_detector.detect(
            mp_image
        )
    )


    # ========================================================
    # GAZE
    # ========================================================

    if face_result.face_landmarks:

        face = face_result.face_landmarks[0]


        gaze = gaze_estimator.estimate(
            face
        )


        if gaze:

            gaze_point = calibration.predict(
                gaze["x"],
                gaze["y"]
            )


            if gaze_point:

                target_x, target_y = gaze_point


                # --------------------------------------------
                # CLAMP
                # --------------------------------------------

                target_x = max(
                    0,
                    min(
                        SCREEN_WIDTH - 1,
                        target_x
                    )
                )


                target_y = max(
                    0,
                    min(
                        SCREEN_HEIGHT - 1,
                        target_y
                    )
                )


                # --------------------------------------------
                # SMOOTH
                # --------------------------------------------

                cursor_x = (
                    cursor_alpha * target_x
                    +
                    (1 - cursor_alpha) * cursor_x
                )


                cursor_y = (
                    cursor_alpha * target_y
                    +
                    (1 - cursor_alpha) * cursor_y
                )


                cursor_x = int(
                    cursor_x
                )


                cursor_y = int(
                    cursor_y
                )


    # ========================================================
    # FIND GAZE TARGET
    # ========================================================

    gaze_target = interaction.find_object(
        cursor_x,
        cursor_y
    )


    # ========================================================
    # UPDATE INTENT
    # ========================================================

    intent_engine.update_gaze_target(
        gaze_target
    )


    # ========================================================
    # HAND
    # ========================================================

    hand_result = (
        hand_detector.detect(
            mp_image
        )
    )


    hand_x = None
    hand_y = None


    if hand_result.hand_landmarks:

        hand = hand_result.hand_landmarks[0]


        hand_data = hand_features.update(
            hand
        )


        normalized_x = hand_data["x"]

        normalized_y = hand_data["y"]


        hand_x = int(
            normalized_x
            *
            SCREEN_WIDTH
        )


        hand_y = int(
            normalized_y
            *
            SCREEN_HEIGHT
        )


        # ====================================================
        # GESTURE
        # ====================================================

        gesture = gesture_recognizer.recognize(
            hand_data
        )


        if gesture:

            print(
                "GESTURE:",
                gesture
            )


            intent = (
                intent_engine.process_gesture(
                    gesture
                )
            )


            if intent:

                action = intent["action"]

                target = intent["target"]


                # ============================================
                # GRAB
                # ============================================

                if action == "GRAB":

                    interaction.grab(
                        target
                    )


                # ============================================
                # RELEASE
                # ============================================

                elif action == "RELEASE":

                    interaction.release()


        # ====================================================
        # MOVE GRABBED OBJECT
        # ====================================================

        if (
            gesture_recognizer.is_pinching
            and
            interaction.grabbed_object
            and
            hand_x is not None
        ):

            interaction.move(
                hand_x,
                hand_y
            )


    else:

        hand_features.reset()


    # ========================================================
    # DRAW AETHER WORLD
    # ========================================================

    world = np.zeros(
        (
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    for obj in interaction.objects:


        # --------------------------------------------
        # GAZE HOVER
        # --------------------------------------------

        if obj == gaze_target:

            cv2.circle(
                world,
                (
                    int(obj.x),
                    int(obj.y)
                ),
                obj.radius + 12,
                (0, 255, 255),
                3
            )


        # --------------------------------------------
        # OBJECT
        # --------------------------------------------

        cv2.circle(
            world,
            (
                int(obj.x),
                int(obj.y)
            ),
            obj.radius,
            (80, 80, 180),
            -1
        )


        # --------------------------------------------
        # GRABBED
        # --------------------------------------------

        if obj == interaction.grabbed_object:

            cv2.circle(
                world,
                (
                    int(obj.x),
                    int(obj.y)
                ),
                obj.radius + 20,
                (0, 255, 0),
                3
            )


        # --------------------------------------------
        # LABEL
        # --------------------------------------------

        cv2.putText(
            world,
            obj.name,
            (
                int(obj.x) - 60,
                int(obj.y) - obj.radius - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # GAZE CURSOR
    # ========================================================

    cv2.circle(
        world,
        (
            cursor_x,
            cursor_y
        ),
        12,
        (0, 255, 255),
        -1
    )


    cv2.circle(
        world,
        (
            cursor_x,
            cursor_y
        ),
        25,
        (0, 255, 255),
        2
    )


    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        world,
        "AETHER",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # ========================================================
    # TARGET STATUS
    # ========================================================

    if gaze_target:

        cv2.putText(
            world,
            f"GAZE: {gaze_target.name}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            world,
            "GAZE: NONE",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # HAND STATUS
    # ========================================================

    if gesture_recognizer.is_pinching:

        cv2.putText(
            world,
            "HAND: PINCH",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            world,
            "HAND: OPEN",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # CURRENT ACTION
    # ========================================================

    if intent_engine.last_action:

        cv2.putText(
            world,
            f"ACTION: {intent_engine.last_action}",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Aether",
        world
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()