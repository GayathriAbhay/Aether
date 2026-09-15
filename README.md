# Aether: Pure Hand-Gesture Human-Computer Interaction (HCI)

Aether is a lightweight, high-precision computer vision system designed to deliver seamless desktop control through pure hand gestures captured via a standard webcam, eliminating the need for external depth sensors, wearable gloves, or gaze-tracking hardware.

---

## Features

* **Sub-Pixel Cursor Tracking**: Maps index-finger landmark coordinates to full-screen dimensions using exponential smoothing (EMA) and direct Win32 API calls for fluid, responsive movement.
* **Precise Gesture State Machine**: 
  * **Single Click**: Quick transient pinch between thumb and index finger.
  * **Double Click**: Rapid double pinch within a defined temporal window.
  * **Drag & Drop**: Sustained pinch hold or fist/grab configuration (curled fingers) to lock down mouse state.
  * **Right Click**: Pinch configuration between thumb and middle finger.
* **Scroll Control**: Vertical displacement tracking between the middle finger and wrist.
* **Hardware Override Protection**: Instantly pauses gesture control if a physical mouse movement is detected, preventing input conflicts.

---

## Repository Structure

```text
aether/
├── aether/
│   ├── camera/
│   │   └── capture.py         # OpenCV camera capture thread
│   ├── hand/
│   │   └── tracker.py         # MediaPipe hand landmarker wrapper
│   ├── gestures/
│   │   └── engine.py          # Landmark processing and gesture classification
│   └── os/
│       └── adapter.py         # Low-level Windows OS mouse injection (pyautogui / Win32)
├── models/
│   └── hand_landmarker.task   # MediaPipe task asset model
└── app.py                     # Main application entry point
