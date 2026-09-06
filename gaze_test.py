import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze import GazeEstimator


# =====================================
# MODEL
# =====================================

MODEL_PATH = "models/face_landmarker.task"


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


# =====================================
# GAZE
# =====================================

gaze_estimator = GazeEstimator()


# =====================================
# CAMERA
# =====================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Could not open camera.")

    exit()


print("Aether gaze tracking started.")
print("Press Q to quit.")


# =====================================
# MAIN LOOP
# =====================================

while True:

    success, frame = camera.read()


    if not success:

        print("Could not read camera.")

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    # =================================
    # RGB
    # =================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =================================
    # FACE DETECTION
    # =================================

    result = detector.detect(
        mp_image
    )


    if result.face_landmarks:

        face = result.face_landmarks[0]


        # =================================
        # GAZE
        # =================================

        gaze = gaze_estimator.estimate(
            face
        )


        if gaze:

            gaze_x = gaze["x"]
            gaze_y = gaze["y"]


            cv2.putText(
                frame,
                f"Gaze X: {gaze_x:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Gaze Y: {gaze_y:.3f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


        # =================================
        # IRIS POINTS
        # =================================

        left_iris = face[468]

        right_iris = face[473]


        width = frame.shape[1]
        height = frame.shape[0]


        left_x = int(
            left_iris.x * width
        )

        left_y = int(
            left_iris.y * height
        )


        right_x = int(
            right_iris.x * width
        )

        right_y = int(
            right_iris.y * height
        )


        # Draw left iris

        cv2.circle(
            frame,
            (left_x, left_y),
            5,
            (0, 255, 255),
            -1
        )


        # Draw right iris

        cv2.circle(
            frame,
            (right_x, right_y),
            5,
            (0, 255, 255),
            -1
        )


        # =================================
        # EYE CENTERS
        # =================================

        left_eye = face[33]

        right_eye = face[263]


        left_eye_x = int(
            left_eye.x * width
        )

        left_eye_y = int(
            left_eye.y * height
        )


        right_eye_x = int(
            right_eye.x * width
        )

        right_eye_y = int(
            right_eye.y * height
        )


        cv2.circle(
            frame,
            (left_eye_x, left_eye_y),
            3,
            (255, 0, 0),
            -1
        )


        cv2.circle(
            frame,
            (right_eye_x, right_eye_y),
            3,
            (255, 0, 0),
            -1
        )


    else:

        cv2.putText(
            frame,
            "NO FACE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =================================
    # DISPLAY
    # =================================

    cv2.imshow(
        "Aether Gaze",
        frame
    )


    # =================================
    # QUIT
    # =================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()