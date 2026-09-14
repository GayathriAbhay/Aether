import time
import ctypes
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

class OSAdapter:
    """Windows Low-Level OS Input Adapter for Gesture Control."""

    def __init__(self, screen_w: int, screen_h: int, alpha: float = 0.35):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.enabled = True

        # Exponential Smoothing (Lower = smooth/delayed, Higher = responsive/raw)
        self.alpha = alpha
        self.smooth_x = float(screen_w / 2)
        self.smooth_y = float(screen_h / 2)

        # Button State Tracking
        self.is_mouse_down = False
        self.last_right_click_time = 0.0

        # Physical Hardware Override
        self.last_physical_pos = pyautogui.position()
        self.override_time = 0.0
        self.override_cooldown = 1.0  # Suspend gestures for 1s if hardware mouse moves

    def set_active(self, status: bool):
        self.enabled = status

    def dispatch(self, target_x: float, target_y: float, left_click: bool, right_click: bool, scroll_amount: int):
        if not self.enabled:
            return

        now = time.time()
        curr_x, curr_y = pyautogui.position()

        # Detect physical mouse movement override
        if abs(curr_x - self.last_physical_pos[0]) > 20 or abs(curr_y - self.last_physical_pos[1]) > 20:
            self.override_time = now

        self.last_physical_pos = (curr_x, curr_y)

        if now - self.override_time < self.override_cooldown:
            return

        # 1. Smooth Coordinates (EMA)
        self.smooth_x = (self.alpha * target_x) + ((1.0 - self.alpha) * self.smooth_x)
        self.smooth_y = (self.alpha * target_y) + ((1.0 - self.alpha) * self.smooth_y)

        clamped_x = max(0, min(self.screen_w - 1, int(self.smooth_x)))
        clamped_y = max(0, min(self.screen_h - 1, int(self.smooth_y)))

        # Move Cursor via direct Win32 API
        ctypes.windll.user32.SetCursorPos(clamped_x, clamped_y)
        self.last_physical_pos = (clamped_x, clamped_y)

        # 2. Left Click / Continuous Dragging
        if left_click and not self.is_mouse_down:
            pyautogui.mouseDown(button='left')
            self.is_mouse_down = True
        elif not left_click and self.is_mouse_down:
            pyautogui.mouseUp(button='left')
            self.is_mouse_down = False

        # 3. Right Click (Debounced 500ms)
        if right_click and (now - self.last_right_click_time > 0.5):
            pyautogui.click(button='right')
            self.last_right_click_time = now

        # 4. Scroll Execution
        if scroll_amount != 0:
            pyautogui.scroll(scroll_amount)