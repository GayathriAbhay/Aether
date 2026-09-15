import time
import numpy as np
from typing import Dict, Any, Tuple

class GestureEngine:
    """Processes 21 hand landmarks into high-precision pointer, click, drag, and double-tap actions."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Dynamic Thresholds
        self.LEFT_CLICK_THRESHOLD = 0.040   # Thumb (4) to Index (8)
        self.RIGHT_CLICK_THRESHOLD = 0.040  # Thumb (4) to Middle (12)
        self.SCROLL_OFFSET_THRESHOLD = 0.12 # Vertical offset between Middle Tip (12) & Wrist (0)
        self.FIST_THRESHOLD = 0.22          # Fingertip to Wrist distance for Fist/Grab

        # Active boundary box
        self.MARGIN_X = 0.18
        self.MARGIN_Y = 0.18

        # Timing and State Variables for Clicks vs Drags
        self.last_pinch_state = False
        self.pinch_start_time = 0.0
        self.last_tap_time = 0.0
        self.double_tap_window = 0.35
        self.tap_hold_threshold = 0.25      # Shorter than this = Tap/Click, Longer = Drag

    def process_landmarks(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        if len(landmarks) < 21:
            return {"hand_detected": False}

        frame_h, frame_w = frame_shape[:2]
        now = time.time()

        # Landmark points
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # 1. Coordinate Mapping (Index Tip to Screen)
        norm_x = (index_tip[0] - self.MARGIN_X) / (1.0 - 2 * self.MARGIN_X)
        norm_y = (index_tip[1] - self.MARGIN_Y) / (1.0 - 2 * self.MARGIN_Y)

        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        raw_screen_x = (1.0 - norm_x) * self.screen_w
        raw_screen_y = norm_y * self.screen_h

        # 2. Left Pinch Detection (Thumb 4 -> Index 8)
        left_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        is_pinched = bool(left_dist < self.LEFT_CLICK_THRESHOLD)

        single_click = False
        double_click = False
        is_dragging = False

        # State transition handling for taps vs drags
        if is_pinched and not self.last_pinch_state:
            # Pinch started down
            self.pinch_start_time = now
        elif not is_pinched and self.last_pinch_state:
            # Pinch released up
            pinch_duration = now - self.pinch_start_time
            if pinch_duration < self.tap_hold_threshold:
                # It was a quick tap! Check for double tap
                time_since_last_tap = now - self.last_tap_time
                if time_since_last_tap < self.double_tap_window:
                    double_click = True
                    self.last_tap_time = 0.0
                else:
                    single_click = True
                    self.last_tap_time = now

        # If pinched and held longer than threshold, it acts as a drag/hold
        if is_pinched and (now - self.pinch_start_time) >= self.tap_hold_threshold:
            is_dragging = True

        self.last_pinch_state = is_pinched

        # 3. Fist / Grab Detection (Alternative drag method)
        fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
        curled_count = sum(
            1 for tip in fingertips 
            if np.linalg.norm(tip[:2] - wrist[:2]) < self.FIST_THRESHOLD
        )
        is_grab = bool(curled_count >= 3)

        # Combine drag triggers (long pinch or fist grab)
        should_drag = is_dragging or is_grab

        # 4. Right Pinch Detection (Thumb 4 -> Middle 12)
        right_dist = np.linalg.norm(thumb_tip[:2] - middle_tip[:2])
        right_click = bool(right_dist < self.RIGHT_CLICK_THRESHOLD)

        # 5. Scroll Detection
        scroll_amount = 0
        middle_wrist_dy = middle_tip[1] - wrist[1]

        if middle_wrist_dy > 0.15:
            scroll_amount = -20
        elif middle_wrist_dy < -0.45:
            scroll_amount = 20

        return {
            "hand_detected": True,
            "target_x": raw_screen_x,
            "target_y": raw_screen_y,
            "single_click": single_click,
            "double_click": double_click,
            "is_pinched": is_pinched,
            "is_dragging": should_drag,
            "right_click": right_click,
            "scroll_amount": scroll_amount,
            "index_px": (int(index_tip[0] * frame_w), int(index_tip[1] * frame_h)),
            "thumb_px": (int(thumb_tip[0] * frame_w), int(thumb_tip[1] * frame_h)),
            "middle_px": (int(middle_tip[0] * frame_w), int(middle_tip[1] * frame_h))
        }