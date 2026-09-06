import cv2
import mediapipe as mp
import numpy as np
import json

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze import GazeEstimator
from calibration import GazeCalibration


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

CALIBRATION_FILE = BASE_DIR / "gaze_calibration.json"


# ============================================================
# LOAD CALIBRATION
# ============================================================

with open(CALIBRATION_FILE, "r") as file:
    calibration_data = json.load(file)


calibration = GazeCalibration()

calibration.matrix = np.array(
    calibration_data["matrix"],
    dtype=np.float64
)


SCREEN_WIDTH = calibration_data["screen_width"]
SCREEN_HEIGHT = calibration_data["screen_height"]


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)


options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,

    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


detector = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# GAZE
# ============================================================

gaze_estimator = GazeEstimator()


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera.")
    exit()


# ============================================================
# CURSOR SMOOTHING
# ============================================================

cursor_x = SCREEN_WIDTH // 2
cursor_y = SCREEN_HEIGHT // 2

cursor_alpha = 0.15


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera.")
        break


    frame = cv2.flip(frame, 1)


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

    result = detector.detect(mp_image)


    if result.face_landmarks:

        face = result.face_landmarks[0]


        # ====================================================
        # GAZE
        # ====================================================

        gaze = gaze_estimator.estimate(face)


        if gaze:

            gaze_x = gaze["x"]
            gaze_y = gaze["y"]


            # ================================================
            # CALIBRATION
            # ================================================

            screen_point = calibration.predict(
                gaze_x,
                gaze_y
            )


            if screen_point:

                target_x, target_y = screen_point


                # ============================================
                # CLAMP
                # ============================================

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


                # ============================================
                # SMOOTH CURSOR
                # ============================================

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


                cursor_x = int(cursor_x)
                cursor_y = int(cursor_y)


    # ========================================================
    # DEBUG WINDOW
    # ========================================================

    debug = frame.copy()


    cv2.putText(
        debug,
        f"Cursor: ({cursor_x}, {cursor_y})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CURSOR REPRESENTATION
    # ========================================================

    cv2.circle(
        debug,
        (
            int(
                cursor_x
                * frame.shape[1]
                / SCREEN_WIDTH
            ),
            int(
                cursor_y
                * frame.shape[0]
                / SCREEN_HEIGHT
            )
        ),
        12,
        (0, 255, 255),
        -1
    )


    cv2.imshow(
        "Aether Gaze Cursor",
        debug
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()