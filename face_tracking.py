import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"


# =========================================
# MEDIAPIPE SETUP
# =========================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
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


# =========================================
# CAMERA
# =========================================

camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:

        print(
            "Could not read camera."
        )

        break


    frame = cv2.flip(
        frame,
        1
    )


    # =====================================
    # BGR → RGB
    # =====================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================
    # MEDIAPIPE IMAGE
    # =====================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================
    # DETECT FACE
    # =====================================

    result = detector.detect(
        mp_image
    )


    # =====================================
    # FACE FOUND
    # =====================================

    if result.face_landmarks:

        face = result.face_landmarks[0]


        # ---------------------------------
        # Left iris
        # ---------------------------------

        left_iris = face[468]


        # ---------------------------------
        # Right iris
        # ---------------------------------

        right_iris = face[473]


        # ---------------------------------
        # Average iris position
        # ---------------------------------

        iris_x = (
            left_iris.x
            +
            right_iris.x
        ) / 2


        iris_y = (
            left_iris.y
            +
            right_iris.y
        ) / 2


        # ---------------------------------
        # Convert to screen coordinates
        # ---------------------------------

        x = int(
            iris_x
            *
            frame.shape[1]
        )


        y = int(
            iris_y
            *
            frame.shape[0]
        )


        # ---------------------------------
        # Draw gaze point
        # ---------------------------------

        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 255, 255),
            -1
        )


        cv2.putText(
            frame,

            "GAZE",

            (x + 15, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 255),

            2
        )


    # =====================================
    # DISPLAY
    # =====================================

    cv2.imshow(
        "Aether Gaze",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()