import numpy as np
from typing import Dict, Any, Tuple

class GestureEngine:
    """Processes 21 hand landmarks into high-precision pointer and click actions."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Dynamic Thresholds (Euclidean distance in normalized space)
        self.LEFT_CLICK_THRESHOLD = 0.040   # Thumb (4) to Index (8)
        self.RIGHT_CLICK_THRESHOLD = 0.040  # Thumb (4) to Middle (12)
        self.SCROLL_OFFSET_THRESHOLD = 0.12 # Vertical offset between Middle Tip (12) & Wrist (0)

        # Active boundary box (margin percentage to reach screen edges easily without straining)
        self.MARGIN_X = 0.18
        self.MARGIN_Y = 0.18

    def process_landmarks(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        if len(landmarks) < 21:
            return {"hand_detected": False}

        frame_h, frame_w = frame_shape[:2]

        # Landmark points
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # 1. Coordinate Mapping (Index Tip to Screen)
        # Remap bounding box [MARGIN, 1 - MARGIN] -> Screen [0, Screen_Size]
        norm_x = (index_tip[0] - self.MARGIN_X) / (1.0 - 2 * self.MARGIN_X)
        norm_y = (index_tip[1] - self.MARGIN_Y) / (1.0 - 2 * self.MARGIN_Y)

        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        # Mirror X coordinate for intuitive webcam control
        raw_screen_x = (1.0 - norm_x) * self.screen_w
        raw_screen_y = norm_y * self.screen_h

        # 2. Left Pinch / Click Detection (Thumb 4 -> Index 8)
        left_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        left_click = bool(left_dist < self.LEFT_CLICK_THRESHOLD)

        # 3. Right Pinch / Click Detection (Thumb 4 -> Middle 12)
        right_dist = np.linalg.norm(thumb_tip[:2] - middle_tip[:2])
        right_click = bool(right_dist < self.RIGHT_CLICK_THRESHOLD)

        # 4. Scroll Detection
        scroll_amount = 0
        middle_wrist_dy = middle_tip[1] - wrist[1]

        # Pointing hand downwards triggers scroll down, pointing up high triggers scroll up
        if middle_wrist_dy > 0.15:
            scroll_amount = -20
        elif middle_wrist_dy < -0.45:
            scroll_amount = 20

        return {
            "hand_detected": True,
            "target_x": raw_screen_x,
            "target_y": raw_screen_y,
            "left_click": left_click,
            "right_click": right_click,
            "scroll_amount": scroll_amount,
            "index_px": (int(index_tip[0] * frame_w), int(index_tip[1] * frame_h)),
            "thumb_px": (int(thumb_tip[0] * frame_w), int(thumb_tip[1] * frame_h)),
            "middle_px": (int(middle_tip[0] * frame_w), int(middle_tip[1] * frame_h))
        }