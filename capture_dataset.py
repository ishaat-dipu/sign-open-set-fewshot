import os
import cv2
import time

# --- Configuration
X1, Y1, X2, Y2 = 150, 100, 450, 400
CAM_INDEX = 4  # DroidCam index

def main():
    cls = input("Class name (e.g., A, V, Unknown): ").strip()
    split = input("Split (train/test/unknown_test): ").strip()

    out_dir = os.path.join("data", split, cls)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {CAM_INDEX}. Check DroidCam connection.")

    print("\n--- RECORDING MODE ---")
    print(f"Class: {cls} | Split: {split}")
    print(f"Saving images to: {out_dir}")
    print("Press SPACE to save frame | 'q' or ESC to exit\n")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Is DroidCam streaming?")
            break

        # 1. Mirror the frame for easier coordination
        frame = cv2.flip(frame, 1)

        # 2. Extract the ROI using  exact coordinates
        roi_frame = frame[Y1:Y2, X1:X2]

        # 3. Create a display copy to draw the UI
        display_frame = frame.copy()

        # Draw the ROI Bounding Box (White)
        cv2.rectangle(display_frame, (X1, Y1), (X2, Y2), (255, 255, 255), 2)
        
        # Labels for the Main Feed
        cv2.putText(display_frame, f"Class: {cls} | Saved: {i}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to save", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 4. Prepare ROI preview (matching  300x300 resize)
        # Note: We resize for the PREVIEW window, but we save the ORIGINAL crop
        roi_preview = cv2.resize(roi_frame, (300, 300))

        # Show both feeds
        cv2.imshow("1. Full Feed (Positioning)", display_frame)
        cv2.imshow("2. ROI Preview (What is saved)", roi_preview)

        # 5. Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == 32:  # SPACE
            timestamp = int(time.time() * 1000)
            fname = f"{cls}_{timestamp}_{i}.jpg"
            path = os.path.join(out_dir, fname)
            
            # SAVE ONLY THE RAW CROPPED ROI
            cv2.imwrite(path, roi_frame)
            print(f"[{i}] Saved ROI: {fname}")
            i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()