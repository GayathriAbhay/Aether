import os
import cv2
import ctypes
import numpy as np

# Suppress native log noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from aether.camera.capture import CameraCapture
from aether.hand.tracker import HandTracker
from aether.gestures.engine import GestureEngine
from aether.os.adapter import OSAdapter

def run_aether():
    print("[+] Initializing Aether pure hand-gesture engine...")

    # Set DPI awareness for high-DPI displays
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    print(f"[+] Active Resolution: {screen_w}x{screen_h}")

    # Initialize Modules
    camera = CameraCapture(0, width=1280, height=720, target_fps=30)
    if not camera.start():
        print("[-] Camera initialization failed.")
        return

    tracker = HandTracker(model_path="models/hand_landmarker.task", min_confidence=0.6)
    gesture_engine = GestureEngine(screen_w, screen_h)
    os_adapter = OSAdapter(screen_w, screen_h, alpha=0.35)

    cv2.namedWindow("Aether Gesture Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aether Gesture Control", 640, 360)

    print("\n" + "="*50)
    print("      AETHER PURE GESTURE HCI ONLINE")
    print("="*50)
    print(" - Move Index Finger Tip : Moves Cursor")
    print(" - Pinch Thumb + Index   : Left Click / Drag")
    print(" - Pinch Thumb + Middle  : Right Click")
    print(" - Press 'c'             : Toggle OS Control ON/OFF")
    print(" - Press 'q' or 'ESC'    : Exit System")
    print("="*50 + "\n")

    try:
        while True:
            frame, fps = camera.get_frame()
            if frame is None:
                continue

            # Flip frame horizontally for mirror visual feedback
            canvas = cv2.flip(frame, 1)

            # Process Hand Tracking
            tracking_res = tracker.process_frame(frame)

            if tracking_res.hand_present:
                gesture_data = gesture_engine.process_landmarks(tracking_res.landmarks, frame.shape)

                if gesture_data["hand_detected"]:
                    # Inject into Windows OS
                    os_adapter.dispatch(
                        gesture_data["target_x"],
                        gesture_data["target_y"],
                        gesture_data["left_click"],
                        gesture_data["right_click"],
                        gesture_data["scroll_amount"]
                    )

                    # Draw Feedback HUD on visual canvas
                    h, w, _ = canvas.shape
                    # Mirror raw coordinates to draw on mirrored canvas
                    idx_x = int((1.0 - tracking_res.landmarks[8][0]) * w)
                    idx_y = int(tracking_res.landmarks[8][1] * h)
                    thumb_x = int((1.0 - tracking_res.landmarks[4][0]) * w)
                    thumb_y = int(tracking_res.landmarks[4][1] * h)

                    # Color shift depending on click status
                    cursor_color = (0, 0, 255) if gesture_data["left_click"] else (0, 255, 0)
                    if gesture_data["right_click"]:
                        cursor_color = (255, 0, 0)

                    cv2.circle(canvas, (idx_x, idx_y), 10, cursor_color, -1)
                    cv2.circle(canvas, (thumb_x, thumb_y), 6, (255, 255, 0), -1)

                    if gesture_data["left_click"]:
                        cv2.line(canvas, (idx_x, idx_y), (thumb_x, thumb_y), (0, 0, 255), 3)

            # Visual HUD
            ctrl_txt = "CONTROL: ON" if os_adapter.enabled else "CONTROL: OFF"
            txt_color = (0, 255, 0) if os_adapter.enabled else (0, 0, 255)
            cv2.putText(canvas, f"FPS: {fps:.1f} | {ctrl_txt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)

            cv2.imshow("Aether Gesture Control", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                os_adapter.set_active(not os_adapter.enabled)
                print(f"[+] OS Control status: {os_adapter.enabled}")

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print("[+] Aether system closed cleanly.")

if __name__ == "__main__":
    run_aether()