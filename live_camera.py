import time
import numpy as np
import torch
import cv2
from mediapipe.tasks.python import vision
from utils_landmarks_tasks import build_hand_landmarker, extract_hand_landmarks_bgr, landmarks_to_features
from models import EmbedNet

# Must match exactly what we used in our capture script!
X1, Y1, X2, Y2 = 150, 100, 450, 400  
CAM_INDEX = 4 # DroidCam index

def draw_viewfinder(img, pt1, pt2, color, length=40, thickness=4):
    """Draws sleek corner brackets instead of a full plain box"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def main():
    # 1. Load the "Brain" and existing training data
    ds = np.load("artifacts/dataset.npz", allow_pickle=True)
    classes = list(ds["classes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load("artifacts/embednet.pt", map_location="cpu")
    emb = EmbedNet(ck["in_dim"], ck["emb_dim"]).to(device)
    emb.load_state_dict(ck["emb"])
    emb.eval()

    # 2. Calculate initial Prototypes (A, B, C, D)
    with torch.no_grad():
        Ztr = emb(torch.from_numpy(ds["Xtr_lm"]).to(device)).cpu().numpy()
    
    protos = np.stack([Ztr[ds["ytr"] == c].mean(axis=0) for c in range(len(classes))])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    
    # 3. Calculate Threshold for Open-Set Rejection (UNKNOWN signs)
    thr = np.percentile((Ztr @ protos.T).max(axis=1), 5)
    print(f"Starting System... UNKNOWN threshold set to: {thr:.3f}")

    # 4. Setup Camera and MediaPipe
    landmarker = build_hand_landmarker("models/hand_landmarker.task", running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(CAM_INDEX)

    # 5. Setup Dynamic Memory variables for Few-Shot Learning
    new_prototypes = []
    new_class_names = []
    prev_time = 0

    print("\n" + "="*40)
    print("🚀 LIVE SYSTEM ACTIVE")
    print("Hold an unlearned sign to the camera until it says UNKNOWN.")
    print("Press 'N' while holding it to instantly teach the system.")
    print("Press 'Q' to quit.")
    print("="*40 + "\n")

    # 6. Main Video Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        roi = frame[Y1:Y2, X1:X2] 
        H, W = frame.shape[:2]
        
        label, score = "NO HAND", 0.0
        ui_color = (150, 150, 150) # Gray

        ts = int(time.time() * 1000)
        lm = extract_hand_landmarks_bgr(roi, landmarker, timestamp_ms=ts)
        
        z_current = None # Store current embedding for teaching

        if lm is not None:
            feat = landmarks_to_features(lm)[None, :].astype(np.float32)
            with torch.no_grad():
                z_current = emb(torch.from_numpy(feat).to(device)).cpu().numpy()
            
            # Combine original Prototypes with any newly taught ones
            if len(new_prototypes) == 0:
                all_protos = protos
            else:
                all_protos = np.vstack([protos, np.vstack(new_prototypes)])
            
            all_labels = classes + new_class_names

            # Compare current hand to all prototypes (Cosine Similarity)
            sims = z_current @ all_protos.T
            max_sim = float(np.max(sims, axis=1)[0])
            
            # Rejection Logic
            if max_sim < thr:
                label = "UNKNOWN"
                ui_color = (0, 140, 255) # Orange for Unknown
            else:
                label = all_labels[int(np.argmax(sims, axis=1)[0])]
                ui_color = (0, 255, 0) # Green for Known
            score = max_sim

        # ================= KEYBOARD CONTROLS =================
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n') and z_current is not None:
            # Simple UI to name the sign in terminal
            new_name = input("\n[TEACH MODE] Enter name for this new sign: ")
            
            # Normalize and save the current embedding as a new Prototype
            z_norm = z_current / (np.linalg.norm(z_current) + 1e-8)
            new_prototypes.append(z_norm)
            new_class_names.append(new_name.upper())
            
            print(f"✅ Success! Sign '{new_name.upper()}' added to memory instantly.")

        elif key == ord('q'):
            break

        # ================= UI RENDERING =================
        # 1. Draw Viewfinder
        draw_viewfinder(frame, (X1, Y1), (X2, Y2), ui_color, length=40, thickness=4)

        # 2. Draw Dark Semi-Transparent Footer
        overlay = frame.copy()
        panel_height = 90
        cv2.rectangle(overlay, (0, H - panel_height), (W, H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 3. Draw Text and Confidence Bar
        if label == "NO HAND":
            cv2.putText(frame, "WAITING FOR GESTURE...", (30, H - 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 180, 180), 2)
        else:
            cv2.putText(frame, f"SIGN: {label}", (30, H - 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, ui_color, 3)
            
            conf_text = f"{max(0, score)*100:.1f}%"
            cv2.putText(frame, conf_text, (W - 140, H - 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            bar_w = 400
            bar_h = 10
            bar_x = 30
            bar_y = H - 25
            fill_w = int((max(0, score) / 1.0) * bar_w)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ui_color, -1)

        # 4. Draw Header and FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(frame, "EDGE AI: PROTOTYPICAL EMBEDNET", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {int(fps)}", (W - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Sign Language Edge AI", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()