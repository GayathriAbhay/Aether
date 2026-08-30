import math
import time


class HandFeatures:

    def __init__(self):

        # ----------------------------------
        # Previous position
        # ----------------------------------

        self.previous_x = None
        self.previous_y = None

        self.previous_time = None


        # ----------------------------------
        # Smoothing
        # ----------------------------------

        self.smoothed_x = None
        self.smoothed_y = None

        self.alpha = 0.35


    def reset(self):

        self.previous_x = None
        self.previous_y = None

        self.previous_time = None

        self.smoothed_x = None
        self.smoothed_y = None


    def distance(self, point1, point2):

        dx = point1.x - point2.x
        dy = point1.y - point2.y

        return math.sqrt(
            dx ** 2 + dy ** 2
        )


    def update(self, hand_landmarks):

        # ==================================
        # LANDMARKS
        # ==================================

        thumb_tip = hand_landmarks[4]

        index_tip = hand_landmarks[8]


        # ==================================
        # PINCH
        # ==================================

        pinch_distance = self.distance(
            thumb_tip,
            index_tip
        )


        # ==================================
        # RAW POSITION
        # ==================================

        raw_x = index_tip.x
        raw_y = index_tip.y


        # ==================================
        # SMOOTH POSITION
        # ==================================

        if self.smoothed_x is None:

            self.smoothed_x = raw_x
            self.smoothed_y = raw_y

        else:

            self.smoothed_x = (
                self.alpha * raw_x
                +
                (1 - self.alpha) * self.smoothed_x
            )

            self.smoothed_y = (
                self.alpha * raw_y
                +
                (1 - self.alpha) * self.smoothed_y
            )


        x = self.smoothed_x
        y = self.smoothed_y


        # ==================================
        # TIME
        # ==================================

        current_time = time.time()


        dx = 0
        dy = 0
        speed = 0


        # ==================================
        # MOVEMENT
        # ==================================

        if self.previous_x is not None:

            dx = x - self.previous_x
            dy = y - self.previous_y


            if self.previous_time is not None:

                dt = current_time - self.previous_time


                if dt > 0:

                    distance = math.sqrt(
                        dx ** 2 + dy ** 2
                    )

                    speed = distance / dt


        # ==================================
        # SAVE STATE
        # ==================================

        self.previous_x = x
        self.previous_y = y

        self.previous_time = current_time


        # ==================================
        # RETURN
        # ==================================

        return {

            "x": x,
            "y": y,

            "dx": dx,
            "dy": dy,

            "speed": speed,

            "pinch_distance": pinch_distance
        }