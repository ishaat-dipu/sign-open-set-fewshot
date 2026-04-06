import os
import csv
import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils_landmarks_tasks import (
    extract_hand_landmarks_bgr,
    landmarks_to_features
)
from models import LandmarkMLP, EmbedNet, TinyCNN

# Note: We need the crop logic here since we are perturbing raw images first
def landmarks_to_bbox_xyxy(lm21x3: np.ndarray, img_w: int, img_h: int, pad: float = 0.20):
    xs = lm21x3[:, 0] * img_w
    ys = lm21x3[:, 1] * img_h
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * bw, x2 + pad * bw
    y1, y2 = y1 - pad * bh, y2 + pad * bh
    x1, x2 = int(max(0, min(img_w - 1, x1))), int(max(0, min(img_w - 1, x2)))
    y1, y2 = int(max(0, min(img_h - 1, y1))), int(max(0, min(img_h - 1, y2)))
    if x2 <= x1 or y2 <= y1: return None
    return (x1, y1, x2, y2)

def crop_resize_roi(img_bgr, bbox_xyxy, out_size=96):
    x1, y1, x2, y2 = bbox_xyxy
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0: return None
    roi = cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

def roi_to_tensor(x_roi_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_roi_uint8).permute(0, 3, 1, 2).float() / 255.0

def apply_perturbation(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clean": return img_bgr.copy()
    if mode == "low_light": return cv2.convertScaleAbs(img_bgr, alpha=0.55, beta=-15)
    if mode == "blur": return cv2.GaussianBlur(img_bgr, (7, 7), sigmaX=1.5)
    if mode == "noise":
        noise = np.random.normal(0, 12, img_bgr.shape).astype(np.float32)
        return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if mode == "rotate":
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 12, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"Unknown mode: {mode}")

def extract_split_with_perturb(root_dir: str, class_names: list, landmarker, perturb="clean", roi_size=96):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    X_lm, X_roi, y = [], [], []
    total, detected = 0, 0
    all_labels, detected_flags, detected_pred_map_idx = [], [], []

    for cname in sorted(os.listdir(root_dir)):
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir) or cname not in class_to_idx: continue
        ci = class_to_idx[cname]
        
        for fname in [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]:
            img = cv2.imread(os.path.join(cdir, fname))
            if img is None: continue

            total += 1
            all_labels.append(ci)
            imgp = apply_perturbation(img, perturb)
            h, w = imgp.shape[:2]

            lm = extract_hand_landmarks_bgr(imgp, landmarker)
            if lm is None:
                detected_flags.append(False)
                detected_pred_map_idx.append(-1)
                continue

            bbox = landmarks_to_bbox_xyxy(lm, w, h)
            if bbox is None:
                detected_flags.append(False)
                detected_pred_map_idx.append(-1)
                continue

            roi = crop_resize_roi(imgp, bbox, out_size=roi_size)
            if roi is None:
                detected_flags.append(False)
                detected_pred_map_idx.append(-1)
                continue

            X_lm.append(landmarks_to_features(lm))
            X_roi.append(roi)
            y.append(ci)
            detected += 1
            detected_flags.append(True)
            detected_pred_map_idx.append(len(y) - 1)

    meta = {
        "total": total, "detected": detected, "det_rate": (detected / max(total, 1)) * 100.0,
        "all_labels": np.array(all_labels, dtype=np.int64),
        "detected_pred_map_idx": np.array(detected_pred_map_idx, dtype=np.int64),
    }

    X_lm = np.array(X_lm, dtype=np.float32)
    X_roi = np.array(X_roi, dtype=np.uint8) if len(X_roi) else np.zeros((0, roi_size, roi_size, 3), dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    return X_lm, X_roi, y, meta

def extract_unknown_with_perturb(root_dir: str, landmarker, perturb="clean", roi_size=96):
    X_lm, X_roi = [], []
    total, detected = 0, 0

    for cname in sorted(os.listdir(root_dir)):
        cdir = os.path.join(root_dir, cname)
        if not os.path.isdir(cdir): continue
        for fname in [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]:
            img = cv2.imread(os.path.join(cdir, fname))
            if img is None: continue
            
            total += 1
            imgp = apply_perturbation(img, perturb)
            h, w = imgp.shape[:2]

            lm = extract_hand_landmarks_bgr(imgp, landmarker)
            if lm is None: continue

            bbox = landmarks_to_bbox_xyxy(lm, w, h)
            if bbox is None: continue

            roi = crop_resize_roi(imgp, bbox, out_size=roi_size)
            if roi is None: continue

            X_lm.append(landmarks_to_features(lm))
            X_roi.append(roi)
            detected += 1

    meta = {"total": total, "detected": detected, "det_rate": (detected / max(total, 1)) * 100.0}
    X_lm = np.array(X_lm, dtype=np.float32)
    X_roi = np.array(X_roi, dtype=np.uint8) if len(X_roi) else np.zeros((0, roi_size, roi_size, 3), dtype=np.uint8)
    return X_lm, X_roi, meta

def end_to_end_accuracy(meta, detected_preds: np.ndarray):
    total = int(meta["total"])
    if total == 0: return 0.0
    labels, idx_map = meta["all_labels"], meta["detected_pred_map_idx"]
    correct = sum(1 for i in range(total) if idx_map[i] >= 0 and detected_preds[idx_map[i]] == labels[i])
    return correct / total

def main():
    # 1. Load Setup
    from mediapipe.tasks.python import vision
    from utils_landmarks_tasks import build_hand_landmarker
    landmarker = build_hand_landmarker("models/hand_landmarker.task", running_mode=vision.RunningMode.IMAGE)

    ds = np.load("artifacts/dataset.npz", allow_pickle=True)
    classes = list(ds["classes"])
    C = len(classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Models
    ck_lm = torch.load("artifacts/landmark_mlp.pt", map_location="cpu")
    lm_model = LandmarkMLP(ck_lm["in_dim"], C).to(device)
    lm_model.load_state_dict(ck_lm["state_dict"])
    lm_model.eval()

    ck_cnn = torch.load("artifacts/tiny_cnn.pt", map_location="cpu")
    cnn_model = TinyCNN(C).to(device)
    cnn_model.load_state_dict(ck_cnn["state_dict"])
    cnn_model.eval()

    ck_emb = torch.load("artifacts/embednet.pt", map_location="cpu")
    emb_model = EmbedNet(ck_emb["in_dim"], ck_emb["emb_dim"]).to(device)
    emb_model.load_state_dict(ck_emb["emb"])
    emb_model.eval()

    # 3. Build Clean Prototypes
    Xtr_lm = torch.from_numpy(ds["Xtr_lm"]).to(device)
    ytr = ds["ytr"]
    with torch.no_grad(): Ztr = emb_model(Xtr_lm).cpu().numpy()
    protos = np.stack([Ztr[ytr == c].mean(axis=0) for c in range(C)])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    perturbations = ["clean", "low_light", "blur", "noise", "rotate"]
    has_unknown = os.path.isdir("data/unknown_test") and len(os.listdir("data/unknown_test")) > 0
    
    results = []
    clean_e2e = {}

    for perturb in perturbations:
        print(f"\n{'='*60}\nEvaluating Perturbation: {perturb.upper()}\n{'='*60}")

        Xte_lm, Xte_roi, yte_det, meta_te = extract_split_with_perturb("data/test", classes, landmarker, perturb=perturb)
        print(f"Known Detection Rate: {meta_te['det_rate']:.1f}%")

        if has_unknown:
            Xunk_lm, Xunk_roi, meta_unk = extract_unknown_with_perturb("data/unknown_test", landmarker, perturb=perturb)
            print(f"Unknown Detection Rate: {meta_unk['det_rate']:.1f}%")
        else:
            Xunk_lm, Xunk_roi, meta_unk = None, None, None

        # A. Landmark-MLP
        pred_lm = np.array([], dtype=np.int64)
        if len(Xte_lm):
            with torch.no_grad(): pred_lm = torch.argmax(lm_model(torch.from_numpy(Xte_lm).to(device)), dim=1).cpu().numpy()
        e2e_lm = end_to_end_accuracy(meta_te, pred_lm)

        # B. TinyCNN
        pred_cnn = np.array([], dtype=np.int64)
        if len(Xte_roi):
            with torch.no_grad(): pred_cnn = torch.argmax(cnn_model(roi_to_tensor(Xte_roi).to(device)), dim=1).cpu().numpy()
        e2e_cnn = end_to_end_accuracy(meta_te, pred_cnn)

        # C. Proposed (Proto)
        pred_prop = np.array([], dtype=np.int64)
        if len(Xte_lm):
            with torch.no_grad(): z = emb_model(torch.from_numpy(Xte_lm).to(device)).cpu().numpy()
            pred_prop = np.argmax(z @ protos.T, axis=1)
        e2e_prop = end_to_end_accuracy(meta_te, pred_prop)

        # Calculate Drops
        if perturb == "clean":
            clean_e2e = {"Landmark-MLP": e2e_lm, "TinyCNN(ROI)": e2e_cnn, "Proposed(Proto)": e2e_prop}

        print(f"\n{'Method':<18} | {'E2E Acc':>8} | {'Drop vs Clean':>14}")
        print("-" * 46)
        print(f"{'Landmark-MLP':<18} | {e2e_lm:8.4f} | {(clean_e2e['Landmark-MLP'] - e2e_lm)*100:13.2f}%")
        print(f"{'TinyCNN(ROI)':<18} | {e2e_cnn:8.4f} | {(clean_e2e['TinyCNN(ROI)'] - e2e_cnn)*100:13.2f}%")
        print(f"{'Proposed(Proto)':<18} | {e2e_prop:8.4f} | {(clean_e2e['Proposed(Proto)'] - e2e_prop)*100:13.2f}%")

        results.extend([
            {"perturbation": perturb, "method": "Landmark-MLP", "det_rate": meta_te["det_rate"], "e2e_acc": e2e_lm},
            {"perturbation": perturb, "method": "TinyCNN(ROI)", "det_rate": meta_te["det_rate"], "e2e_acc": e2e_cnn},
            {"perturbation": perturb, "method": "Proposed(Proto)", "det_rate": meta_te["det_rate"], "e2e_acc": e2e_prop},
        ])

    with open("artifacts/robustness_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", "method", "det_rate", "e2e_acc"])
        writer.writeheader()
        writer.writerows(results)
    print("\n Saved detailed results to artifacts/robustness_results.csv")

if __name__ == "__main__":
    main()