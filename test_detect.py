import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"D:\sign-open-set-fewshot\models\hand_landmarker.task"
IMG_PATH = r"D:\sign-open-set-fewshot\test.jpg"  # change to your real image

# Build landmarker
base = python.BaseOptions(model_asset_path=MODEL_PATH)
opts = vision.HandLandmarkerOptions(
    base_options=base,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE,
)
landmarker = vision.HandLandmarker.create_from_options(opts)

# Load image
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMG_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#  Correct MPImage construction for  version
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

# Detect
res = landmarker.detect(mp_image)

if not res.hand_landmarks:
    print("No hand detected  (try a clearer hand image)")
else:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]], dtype=np.float32)
    print("Landmarks ", pts.shape)
    print("Wrist:", pts[0])