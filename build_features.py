import os
import cv2
import numpy as np
from tqdm import tqdm
from mediapipe.tasks.python import vision
from utils_landmarks_tasks import build_hand_landmarker, extract_hand_landmarks_bgr, landmarks_to_features

def load_split(root_dir, landmarker, roi_size=96):
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    X_lm, X_roi, y = [], [], []
    total, detected = 0, 0

    for ci, cname in enumerate(class_names):
        cdir = os.path.join(root_dir, cname)
        files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        for fname in tqdm(files, desc=f"Loading {root_dir}/{cname}"):
            total += 1
            img = cv2.imread(os.path.join(cdir, fname))
            if img is None: continue

            # 1. Resize for TinyCNN baseline
            roi_img = cv2.resize(img, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

            # 2. Extract Landmarks
            lm = extract_hand_landmarks_bgr(img, landmarker)
            if lm is None: continue

            feat = landmarks_to_features(lm)
            
            X_lm.append(feat)
            X_roi.append(roi_rgb)
            y.append(ci)
            detected += 1

    X_lm = np.array(X_lm, dtype=np.float32)
    X_roi = np.array(X_roi, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    det_rate = (detected / max(total, 1)) * 100.0

    return X_lm, X_roi, y, class_names, det_rate, total, detected

def main():
    os.makedirs("artifacts", exist_ok=True)
    
    # Use IMAGE mode for static dataset building
    landmarker = build_hand_landmarker("models/hand_landmarker.task", running_mode=vision.RunningMode.IMAGE)

    print("--- Processing Train & Test Splits ---")
    Xtr_lm, Xtr_roi, ytr, classes, tr_det_rate, tr_tot, tr_det = load_split("data/train", landmarker)
    Xte_lm, Xte_roi, yte, classes2, te_det_rate, te_tot, te_det = load_split("data/test", landmarker)
    
    assert classes == classes2, "Train/test classes mismatch!"

    save_dict = {
        "Xtr_lm": Xtr_lm, "Xtr_roi": Xtr_roi, "ytr": ytr,
        "Xte_lm": Xte_lm, "Xte_roi": Xte_roi, "yte": yte,
        "classes": np.array(classes, dtype=object),
        "tr_det_rate": tr_det_rate, "te_det_rate": te_det_rate
    }

    # Process Unknown/Open-Set data
    if os.path.isdir("data/unknown_test") and len(os.listdir("data/unknown_test")) > 0:
        print("\n--- Processing Unknown/Open-Set Split ---")
        Xunk_lm, Xunk_roi, _, _, unk_det_rate, unk_tot, unk_det = load_split("data/unknown_test", landmarker)
        save_dict.update({"Xunk_lm": Xunk_lm, "Xunk_roi": Xunk_roi, "unk_det_rate": unk_det_rate})
        print(f"Unknown Detection Rate: {unk_det_rate:.1f}% ({unk_det}/{unk_tot})")

    np.savez("artifacts/dataset.npz", **save_dict)
    print(f"\n✅ Saved dataset.npz. Train Det Rate: {tr_det_rate:.1f}%, Test Det Rate: {te_det_rate:.1f}%")

if __name__ == "__main__":
    main()