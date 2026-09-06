import math
import numpy as np


class GazeEstimator:

    def __init__(self):

        # Smoothed feature values
        self.smoothed_features = None

        # Lower = smoother, higher = more responsive
        self.alpha = 0.25

        # MediaPipe Face Landmarker iris landmarks
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]

    # ---------------------------------------------------------
    # DISTANCE
    # ---------------------------------------------------------

    def distance(self, p1, p2):

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # ---------------------------------------------------------
    # IRIS CENTER
    # ---------------------------------------------------------

    def iris_center(
        self,
        face,
        indices
    ):

        xs = []
        ys = []

        for index in indices:

            xs.append(face[index].x)
            ys.append(face[index].y)

        return (
            sum(xs) / len(xs),
            sum(ys) / len(ys)
        )

    # ---------------------------------------------------------
    # NORMALIZE IRIS INSIDE EYE
    # ---------------------------------------------------------

    def normalize_eye(
        self,
        iris_x,
        iris_y,
        outer,
        inner,
        top,
        bottom
    ):

        eye_width = self.distance(
            outer,
            inner
        )

        eye_height = self.distance(
            top,
            bottom
        )

        if eye_width < 1e-6:
            return None

        if eye_height < 1e-6:
            return None

        # Horizontal position
        horizontal = (
            iris_x - inner.x
        ) / eye_width

        # Vertical position
        vertical = (
            iris_y - top.y
        ) / eye_height

        return (
            horizontal,
            vertical
        )

    # ---------------------------------------------------------
    # SMOOTH FEATURES
    # ---------------------------------------------------------

    def smooth(self, features):

        features = np.array(
            features,
            dtype=np.float64
        )

        if self.smoothed_features is None:

            self.smoothed_features = features

        else:

            self.smoothed_features = (
                self.alpha * features
                +
                (1 - self.alpha)
                * self.smoothed_features
            )

        return self.smoothed_features.tolist()

    # ---------------------------------------------------------
    # MAIN GAZE FEATURE EXTRACTION
    # ---------------------------------------------------------

    def extract_features(self, face):

        # -----------------------------------------------------
        # IRIS CENTERS
        # -----------------------------------------------------

        left_iris_x, left_iris_y = self.iris_center(
            face,
            self.left_iris_indices
        )

        right_iris_x, right_iris_y = self.iris_center(
            face,
            self.right_iris_indices
        )

        # -----------------------------------------------------
        # LEFT EYE
        # -----------------------------------------------------

        left_eye = self.normalize_eye(

            left_iris_x,
            left_iris_y,

            # outer
            face[33],

            # inner
            face[133],

            # top
            face[159],

            # bottom
            face[145]
        )

        # -----------------------------------------------------
        # RIGHT EYE
        # -----------------------------------------------------

        right_eye = self.normalize_eye(

            right_iris_x,
            right_iris_y,

            # outer
            face[263],

            # inner
            face[362],

            # top
            face[386],

            # bottom
            face[374]
        )

        if left_eye is None or right_eye is None:
            return None

        left_x, left_y = left_eye
        right_x, right_y = right_eye

        # -----------------------------------------------------
        # COMBINED FEATURES
        # -----------------------------------------------------

        average_x = (
            left_x +
            right_x
        ) / 2

        average_y = (
            left_y +
            right_y
        ) / 2

        # Difference between eyes
        # Useful for detecting asymmetry
        eye_difference_x = (
            left_x -
            right_x
        )

        eye_difference_y = (
            left_y -
            right_y
        )

        features = [
            left_x,
            left_y,

            right_x,
            right_y,

            average_x,
            average_y,

            eye_difference_x,
            eye_difference_y
        ]

        return self.smooth(features)

    # ---------------------------------------------------------
    # SIMPLE GAZE POINT
    # ---------------------------------------------------------

    def estimate(self, face):

        features = self.extract_features(face)

        if features is None:
            return None

        return {
            "x": features[4],
            "y": features[5],

            "features": features
        }