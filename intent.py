import time


class IntentEngine:

    def __init__(self):

        self.current_target = None

        self.previous_target = None

        self.is_grabbing = False

        self.last_action = None

        self.last_action_time = 0

        self.action_cooldown = 0.2


    # =========================================================
    # FIND WHETHER WE CAN TRIGGER AN ACTION
    # =========================================================

    def can_trigger(self):

        current_time = time.time()

        return (
            current_time - self.last_action_time
            >= self.action_cooldown
        )


    # =========================================================
    # UPDATE GAZE TARGET
    # =========================================================

    def update_gaze_target(self, target):

        self.previous_target = self.current_target

        self.current_target = target


    # =========================================================
    # PROCESS HAND EVENT
    # =========================================================

    def process_gesture(self, gesture):

        if gesture is None:

            return None


        # =====================================================
        # PINCH START
        # =====================================================

        if gesture == "PINCH_START":

            if self.current_target is not None:

                self.is_grabbing = True

                self.last_action = "GRAB"

                self.last_action_time = time.time()

                return {
                    "action": "GRAB",
                    "target": self.current_target
                }

            else:

                self.last_action = "PINCH"

                self.last_action_time = time.time()

                return {
                    "action": "PINCH",
                    "target": None
                }


        # =====================================================
        # PINCH END
        # =====================================================

        if gesture == "PINCH_END":

            if self.is_grabbing:

                target = self.current_target

                self.is_grabbing = False

                self.last_action = "RELEASE"

                self.last_action_time = time.time()

                return {
                    "action": "RELEASE",
                    "target": target
                }


        # =====================================================
        # SWIPE
        # =====================================================

        if gesture.startswith("SWIPE_"):

            self.last_action = gesture

            self.last_action_time = time.time()

            return {
                "action": gesture,
                "target": self.current_target
            }


        return None