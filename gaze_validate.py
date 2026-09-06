import cv2
import json
import math
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze import GazeEstimator
from calibration import GazeCalibration
from screen import get_screen_size


# ============================================================
# SCREEN
# ============================================================

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()


# ============================================================
# LOAD CALIBRATION
# ============================================================

try:

    with open(
        "gaze_calibration.json",
        "r"
    ) as file:

        data = json.load(file)

except FileNotFoundError:

    print(
        "ERROR: gaze_calibration.json "
        "not found."
    )

    print(
        "Run calibrate.py first."
    )

    exit()


# ============================================================
# CALIBRATION MODEL
# ============================================================

model = GazeCalibration()

model.weights_x = np.array(
    data["weights_x"],
    dtype=np.float64
)

model.weights_y = np.array(
    data["weights_y"],
    dtype=np.float64
)


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(

    base_options=base_options,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False,

    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# GAZE
# ============================================================

gaze = GazeEstimator()


# ============================================================
# VALIDATION POINTS
# ============================================================

points = [

    (
        int(SCREEN_WIDTH * 0.10),
        int(SCREEN_HEIGHT * 0.10)
    ),

    (
        int(SCREEN_WIDTH * 0.50),
        int(SCREEN_HEIGHT * 0.10)
    ),

    (
        int(SCREEN_WIDTH * 0.90),
        int(SCREEN_HEIGHT * 0.10)
    ),

    (
        int(SCREEN_WIDTH * 0.10),
        int(SCREEN_HEIGHT * 0.50)
    ),

    (
        int(SCREEN_WIDTH * 0.50),
        int(SCREEN_HEIGHT * 0.50)
    ),

    (
        int(SCREEN_WIDTH * 0.90),
        int(SCREEN_HEIGHT * 0.50)
    ),

    (
        int(SCREEN_WIDTH * 0.10),
        int(SCREEN_HEIGHT * 0.90)
    ),

    (
        int(SCREEN_WIDTH * 0.50),
        int(SCREEN_HEIGHT * 0.90)
    ),

    (
        int(SCREEN_WIDTH * 0.90),
        int(SCREEN_HEIGHT * 0.90)
    )
]


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ============================================================
# WINDOW
# ============================================================

WINDOW = "AETHER VALIDATION"

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# RESULTS
# ============================================================

errors = []


# ============================================================
# VALIDATION
# ============================================================

for index, target in enumerate(points):

    target_x, target_y = target

    samples = []

    print(
        f"Validation point "
        f"{index + 1}/9"
    )

    # --------------------------------------------------------
    # STABILIZATION
    # --------------------------------------------------------

    start_time = time.time() if False else 0

    for _ in range(30):

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        detector.detect(
            image
        )

        screen = np.zeros(
            (
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        cv2.circle(
            screen,
            target,
            20,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            screen,

            "Look at the dot",

            (
                50,
                70
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (255, 255, 255),

            2
        )

        cv2.imshow(
            WINDOW,
            screen
        )

        if cv2.waitKey(1) & 0xFF == 27:

            cap.release()

            cv2.destroyAllWindows()

            exit()

    # --------------------------------------------------------
    # COLLECT
    # --------------------------------------------------------

    while len(samples) < 30:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(
            image
        )

        if len(result.face_landmarks) == 0:

            continue

        face = result.face_landmarks[0]

        features = gaze.extract_features(
            face
        )

        if features is None:

            continue

        predicted = model.predict(
            features
        )

        if predicted is None:

            continue

        samples.append(
            predicted
        )

        # ----------------------------------------------------
        # SCREEN
        # ----------------------------------------------------

        screen = np.zeros(
            (
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # Target
        cv2.circle(
            screen,
            target,
            20,
            (255, 255, 255),
            -1
        )

        # Predicted gaze
        px = int(
            max(
                0,
                min(
                    SCREEN_WIDTH - 1,
                    predicted[0]
                )
            )
        )

        py = int(
            max(
                0,
                min(
                    SCREEN_HEIGHT - 1,
                    predicted[1]
                )
            )
        )

        cv2.circle(
            screen,
            (px, py),
            12,
            (255, 255, 255),
            -1
        )

        cv2.imshow(
            WINDOW,
            screen
        )

        if cv2.waitKey(1) & 0xFF == 27:

            cap.release()

            cv2.destroyAllWindows()

            exit()

    # --------------------------------------------------------
    # AVERAGE PREDICTION
    # --------------------------------------------------------

    predicted_x = np.mean(
        [
            p[0]
            for p in samples
        ]
    )

    predicted_y = np.mean(
        [
            p[1]
            for p in samples
        ]
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    error = math.sqrt(

        (
            predicted_x
            -
            target_x
        ) ** 2

        +

        (
            predicted_y
            -
            target_y
        ) ** 2
    )

    errors.append(
        error
    )

    print(
        f"Target: "
        f"({target_x}, {target_y})"
    )

    print(
        f"Predicted: "
        f"({predicted_x:.1f}, "
        f"{predicted_y:.1f})"
    )

    print(
        f"Error: "
        f"{error:.1f}px"
    )

    print()


# ============================================================
# FINAL RESULTS
# ============================================================

errors = np.array(
    errors
)


print()
print(
    "======================================"
)

print(
    " AETHER GAZE VALIDATION"
)

print(
    "======================================"
)

print(
    f"Mean error: "
    f"{np.mean(errors):.2f}px"
)

print(
    f"Median error: "
    f"{np.median(errors):.2f}px"
)

print(
    f"95th percentile: "
    f"{np.percentile(errors, 95):.2f}px"
)

print(
    "======================================"
)


cap.release()

cv2.destroyAllWindows()