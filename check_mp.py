import mediapipe as mp
import inspect, sys

print("mp module file:", getattr(mp, "__file__", None))
print("mp version:", getattr(mp, "__version__", None))
print("has solutions:", hasattr(mp, "solutions"))
print("dir contains 'solutions'?:", "solutions" in dir(mp))
print("python:", sys.executable)