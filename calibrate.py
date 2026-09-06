import cv2
import json
import time
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

print(
    f"Screen resolution: "
    f"{SCREEN_WIDTH} x {SCREEN_HEIGHT}"
)


# ============================================================
# MEDIAPIPE FACE LANDMARKER
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
# GAZE ESTIMATOR
# ============================================================

gaze_estimator = GazeEstimator()


# ============================================================
# CALIBRATION GRID
# ============================================================

margin_x = int(
    SCREEN_WIDTH * 0.10
)

margin_y = int(
    SCREEN_HEIGHT * 0.10
)


points = [

    # Top
    (
        margin_x,
        margin_y
    ),

    (
        SCREEN_WIDTH // 2,
        margin_y
    ),

    (
        SCREEN_WIDTH - margin_x,
        margin_y
    ),

    # Middle
    (
        margin_x,
        SCREEN_HEIGHT // 2
    ),

    (
        SCREEN_WIDTH // 2,
        SCREEN_HEIGHT // 2
    ),

    (
        SCREEN_WIDTH - margin_x,
        SCREEN_HEIGHT // 2
    ),

    # Bottom
    (
        margin_x,
        SCREEN_HEIGHT - margin_y
    ),

    (
        SCREEN_WIDTH // 2,
        SCREEN_HEIGHT - margin_y
    ),

    (
        SCREEN_WIDTH - margin_x,
        SCREEN_HEIGHT - margin_y
    )
]


# ============================================================
# WINDOW
# ============================================================

WINDOW = "AETHER CALIBRATION"

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
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ============================================================
# DATA
# ============================================================

all_features = []

all_screen_points = []


# ============================================================
# SETTINGS
# ============================================================

WAIT_TIME = 1.0

SAMPLES = 60


# ============================================================
# CALIBRATION
# ============================================================

for point_number, point in enumerate(points):

    target_x, target_y = point

    print(
        f"Calibration point "
        f"{point_number + 1}/9"
    )

    # --------------------------------------------------------
    # STABILIZATION PERIOD
    # --------------------------------------------------------

    start_time = time.time()

    while (
        time.time() - start_time
        <
        WAIT_TIME
    ):

        ret, camera_frame = cap.read()

        if not ret:
            continue

        camera_frame = cv2.flip(
            camera_frame,
            1
        )

        rgb = cv2.cvtColor(
            camera_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Run detection to allow
        # gaze smoothing to settle

        detector.detect(
            mp_image
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
            point,
            18,
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

        cv2.putText(
            screen,

            f"Point {point_number + 1}/9",

            (
                50,
                120
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (255, 255, 255),

            2
        )

        cv2.imshow(
            WINDOW,
            screen
        )

        key = cv2.waitKey(1)

        if key == 27:

            cap.release()

            cv2.destroyAllWindows()

            exit()

    # --------------------------------------------------------
    # COLLECT SAMPLES
    # --------------------------------------------------------

    collected = []

    while len(collected) < SAMPLES:

        ret, camera_frame = cap.read()

        if not ret:
            continue

        camera_frame = cv2.flip(
            camera_frame,
            1
        )

        rgb = cv2.cvtColor(
            camera_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(
            mp_image
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

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
            point,
            18,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            screen,

            f"Collecting "
            f"{len(collected)}/{SAMPLES}",

            (
                50,
                70
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (255, 255, 255),

            2
        )

        cv2.imshow(
            WINDOW,
            screen
        )

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        if len(result.face_landmarks) > 0:

            face = result.face_landmarks[0]

            features = (
                gaze_estimator
                .extract_features(face)
            )

            if features is not None:

                collected.append(
                    features
                )

        key = cv2.waitKey(1)

        if key == 27:

            cap.release()

            cv2.destroyAllWindows()

            exit()

    # --------------------------------------------------------
    # OUTLIER FILTERING
    # --------------------------------------------------------

    collected = np.array(
        collected,
        dtype=np.float64
    )

    median = np.median(
        collected,
        axis=0
    )

    distances = np.linalg.norm(
        collected - median,
        axis=1
    )

    threshold = np.percentile(
        distances,
        80
    )

    filtered = collected[
        distances <= threshold
    ]

    print(
        f"Samples: {len(collected)}"
    )

    print(
        f"Filtered: {len(filtered)}"
    )

    # --------------------------------------------------------
    # REPRESENTATIVE FEATURE VECTOR
    # --------------------------------------------------------

    representative = np.median(
        filtered,
        axis=0
    )

    all_features.append(
        representative.tolist()
    )

    all_screen_points.append(
        [
            target_x,
            target_y
        ]
    )


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print(
    "Training gaze calibration model..."
)


model = GazeCalibration()

model.fit(
    all_features,
    all_screen_points
)


# ============================================================
# SAVE MODEL
# ============================================================

data = {

    "screen_width":
        SCREEN_WIDTH,

    "screen_height":
        SCREEN_HEIGHT,

    "features":
        all_features,

    "screen_points":
        all_screen_points,

    "weights_x":
        model.weights_x.tolist(),

    "weights_y":
        model.weights_y.tolist()
}


with open(
    "gaze_calibration.json",
    "w"
) as file:

    json.dump(
        data,
        file,
        indent=4
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print(
    "======================================"
)

print(
    " AETHER CALIBRATION COMPLETE"
)

print(
    "======================================"
)

print()
print(
    "Saved: gaze_calibration.json"
)