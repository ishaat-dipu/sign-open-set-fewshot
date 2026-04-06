import time
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------
# Build landmarker ONCE
# ----------------------------
def build_hand_landmarker(
    model_path: str,
    num_hands: int = 1,
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """
    For webcam/stream use VIDEO mode (recommended).
    For offline images use IMAGE mode.
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        running_mode=running_mode,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


# ----------------------------
# Extract landmarks
# ----------------------------
def extract_hand_landmarks_bgr(img_bgr, landmarker, timestamp_ms: int | None = None):
    """
    Returns:
      pts: (21,3) float32 in normalized coords OR None
    Notes:
      - If landmarker is VIDEO mode, you must provide timestamp_ms.
      - If landmarker is IMAGE mode, timestamp_ms is ignored.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Bulletproof Mode Check:
    # If a timestamp is passed, we safely assume VIDEO mode.
    if timestamp_ms is not None:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    return pts


# ----------------------------
# Feature engineering
# ----------------------------
def landmarks_to_features(lm21x3: np.ndarray) -> np.ndarray:
    """
    (21,3) -> 83D feature:
      - 63 wrist-centered normalized coords
      - 20 bone lengths
    """
    wrist = lm21x3[0].copy()
    centered = lm21x3 - wrist

    # scale normalize (distance wrist -> middle MCP)
    ref = np.linalg.norm(centered[9, :2]) + 1e-6
    centered = centered / ref

    feat = centered.flatten()  # 63 features

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    lengths = np.array(
        [np.linalg.norm(centered[b, :2] - centered[a, :2]) for a, b in edges],
        dtype=np.float32
    )

    return np.concatenate([feat, lengths], axis=0).astype(np.float32)