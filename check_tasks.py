from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"D:\sign-open-set-fewshot\models\hand_landmarker.task"

base = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE,
)

landmarker = vision.HandLandmarker.create_from_options(options)

print("HandLandmarker loaded successfully ")