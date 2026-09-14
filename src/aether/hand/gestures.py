import numpy as np
from enum import Enum, auto
from typing import Optional

class GestureType(Enum):
    NONE = auto()
    POINT = auto()
    PINCH = auto()
    OPEN_PALM = auto()

class GestureState(Enum):
    NONE = auto()
    CANDIDATE = auto()
    CONFIRMED = auto()

class GestureStateMachine:
    """
    Temporal gesture state machine with debouncing to prevent trigger flutter.
    """

    INDEX_TIP = 8
    THUMB_TIP = 4
    MIDDLE_TIP = 12

    def __init__(self, confirm_frames: int = 3, pinch_threshold: float = 0.05):
        self.confirm_frames = confirm_frames
        self.pinch_threshold = pinch_threshold

        self.current_candidate = GestureType.NONE
        self.candidate_count = 0
        self.active_gesture = GestureType.NONE

    def update(self, hand_landmarks: np.ndarray, hand_present: bool) -> GestureType:
        if not hand_present or len(hand_landmarks) < 21:
            self._reset()
            return GestureType.NONE

        raw_gesture = self._classify_frame(hand_landmarks)

        # State transition logic
        if raw_gesture == self.current_candidate:
            self.candidate_count += 1
        else:
            self.current_candidate = raw_gesture
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            self.active_gesture = self.current_candidate

        return self.active_gesture

    def _classify_frame(self, lm: np.ndarray) -> GestureType:
        # Distance between thumb tip and index tip
        thumb_tip = lm[self.THUMB_TIP]
        index_tip = lm[self.INDEX_TIP]
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)

        if pinch_dist < self.pinch_threshold:
            return GestureType.PINCH

        # Extended index checking for Pointing
        if index_tip[1] < lm[6][1] and lm[self.MIDDLE_TIP][1] > lm[10][1]:
            return GestureType.POINT

        return GestureType.NONE

    def _reset(self):
        self.current_candidate = GestureType.NONE
        self.candidate_count = 0
        self.active_gesture = GestureType.NONE