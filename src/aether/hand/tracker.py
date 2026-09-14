import os
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

@dataclass
class HandTrackerResult:
    hand_present: bool
    landmarks: np.ndarray        # Shape (21, 3) normalized [0, 1]
    world_landmarks: np.ndarray  # Shape (21, 3) metric 3D space

class HandTracker:
    """MediaPipe Tasks Hand Landmarker wrapper."""

    def __init__(self, model_path: str = "models/hand_landmarker.task", min_confidence: float = 0.6):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at '{model_path}'. Ensure hand_landmarker.task exists.")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process_frame(self, frame_bgr: np.ndarray) -> HandTrackerResult:
        self.timestamp_ms += 33  # Approx 30 FPS step
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return HandTrackerResult(hand_present=False, landmarks=np.empty((0, 3)), world_landmarks=np.empty((0, 3)))

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]], dtype=np.float32)

        return HandTrackerResult(hand_present=True, landmarks=landmarks, world_landmarks=world_landmarks)

    def close(self):
        self.landmarker.close()