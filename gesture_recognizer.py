import time


class GestureRecognizer:

    def __init__(self):

        # =================================
        # PINCH STATE
        # =================================

        self.pinch_threshold = 0.05

        self.release_threshold = 0.07

        self.is_pinching = False


        # =================================
        # SWIPE
        # =================================

        self.swipe_speed_threshold = 0.8

        self.swipe_movement_threshold = 0.015

        self.last_swipe_time = 0

        self.swipe_cooldown = 0.5


    # =====================================
    # PINCH
    # =====================================

    def detect_pinch(self, distance):

        if not self.is_pinching:

            if distance < self.pinch_threshold:

                self.is_pinching = True

                return "PINCH_START"


        else:

            if distance > self.release_threshold:

                self.is_pinching = False

                return "PINCH_END"


        return None


    # =====================================
    # SWIPE
    # =====================================

    def detect_swipe(
        self,
        dx,
        dy,
        speed
    ):

        current_time = time.time()


        # Cooldown
        if (
            current_time
            -
            self.last_swipe_time
            <
            self.swipe_cooldown
        ):
            return None


        # Speed
        if speed < self.swipe_speed_threshold:
            return None


        # Movement
        if (
            abs(dx) < self.swipe_movement_threshold
            and
            abs(dy) < self.swipe_movement_threshold
        ):
            return None


        # Horizontal
        if abs(dx) > abs(dy):

            if dx > 0:

                gesture = "SWIPE_RIGHT"

            else:

                gesture = "SWIPE_LEFT"


        # Vertical
        else:

            if dy > 0:

                gesture = "SWIPE_DOWN"

            else:

                gesture = "SWIPE_UP"


        self.last_swipe_time = current_time

        return gesture


    # =====================================
    # MAIN
    # =====================================

    def recognize(self, features):

        pinch_distance = features["pinch_distance"]

        dx = features["dx"]

        dy = features["dy"]

        speed = features["speed"]


        # ---------------------------------
        # Pinch
        # ---------------------------------

        event = self.detect_pinch(
            pinch_distance
        )

        if event:

            return event


        # ---------------------------------
        # Swipe
        # ---------------------------------

        event = self.detect_swipe(
            dx,
            dy,
            speed
        )

        if event:

            return event


        return None