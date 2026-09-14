import cv2
import time
import threading
from typing import Tuple, Optional

class CameraCapture:
    """Threaded OpenCV camera capture pipeline for low-latency frame streaming."""

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Performance counters
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            with self.lock:
                self.frame = frame
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

    def get_frame(self) -> Tuple[Optional[cv2.Mat], float]:
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.fps

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()