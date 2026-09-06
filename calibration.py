import numpy as np


class GazeCalibration:

    def __init__(self):

        self.weights_x = None
        self.weights_y = None

    # ========================================================
    # FEATURE TRANSFORMATION
    # ========================================================

    def transform(self, features):

        features = np.array(
            features,
            dtype=np.float64
        )

        result = []

        # ----------------------------------------------------
        # ORIGINAL FEATURES
        # ----------------------------------------------------

        result.extend(
            features.tolist()
        )

        # ----------------------------------------------------
        # SQUARED FEATURES
        # ----------------------------------------------------

        result.extend(
            (features ** 2).tolist()
        )

        # ----------------------------------------------------
        # INTERACTION FEATURES
        # ----------------------------------------------------

        for i in range(len(features)):

            for j in range(
                i + 1,
                len(features)
            ):

                result.append(
                    features[i]
                    *
                    features[j]
                )

        # ----------------------------------------------------
        # BIAS
        # ----------------------------------------------------

        result.append(1.0)

        return np.array(
            result,
            dtype=np.float64
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        gaze_features,
        screen_points
    ):

        X = []

        Yx = []
        Yy = []

        for features, screen in zip(
            gaze_features,
            screen_points
        ):

            transformed = self.transform(
                features
            )

            X.append(
                transformed
            )

            Yx.append(
                screen[0]
            )

            Yy.append(
                screen[1]
            )

        X = np.array(
            X,
            dtype=np.float64
        )

        Yx = np.array(
            Yx,
            dtype=np.float64
        )

        Yy = np.array(
            Yy,
            dtype=np.float64
        )

        # ----------------------------------------------------
        # LEAST SQUARES
        # ----------------------------------------------------

        self.weights_x = np.linalg.lstsq(
            X,
            Yx,
            rcond=None
        )[0]

        self.weights_y = np.linalg.lstsq(
            X,
            Yy,
            rcond=None
        )[0]

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features
    ):

        if (
            self.weights_x is None
            or
            self.weights_y is None
        ):
            return None

        transformed = self.transform(
            features
        )

        x = (
            transformed
            @
            self.weights_x
        )

        y = (
            transformed
            @
            self.weights_y
        )

        return (
            float(x),
            float(y)
        )