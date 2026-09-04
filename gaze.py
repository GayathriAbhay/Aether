import math


class GazeEstimator:

    def __init__(self):

        self.previous_x = None
        self.previous_y = None

        self.smoothed_x = None
        self.smoothed_y = None

        # Smoothing factor
        self.alpha = 0.2


    def smooth(self, x, y):

        if self.smoothed_x is None:

            self.smoothed_x = x
            self.smoothed_y = y

        else:

            self.smoothed_x = (
                self.alpha * x
                +
                (1 - self.alpha)
                * self.smoothed_x
            )

            self.smoothed_y = (
                self.alpha * y
                +
                (1 - self.alpha)
                * self.smoothed_y
            )

        return (
            self.smoothed_x,
            self.smoothed_y
        )


    def estimate(self, face_landmarks):

        # -------------------------------------
        # Iris landmarks
        # -------------------------------------

        left_iris = face_landmarks[468]
        right_iris = face_landmarks[473]


        # -------------------------------------
        # Calculate average iris position
        # -------------------------------------

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


        # -------------------------------------
        # Smooth
        # -------------------------------------

        x, y = self.smooth(
            iris_x,
            iris_y
        )


        return {
            "x": x,
            "y": y
        }