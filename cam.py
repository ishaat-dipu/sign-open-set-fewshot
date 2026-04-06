import cv2

def main():
    # --- 1. SETUP ---
    # Using 3 for DroidCam based on previous logs
    cap = cv2.VideoCapture(4) 
    
    if not cap.isOpened():
        print("Error: Could not open DroidCam. Check index (0, 1, 2, or 3).")
        return

    # Bounding Box (Adjust these to perfectly frame hand)
    X1, Y1, X2, Y2 = 150, 100, 450, 400

    print("--- NATURAL TEST MODE ---")
    print("1. Ensure your hand fills ~70% of the White Box.")
    print("2. Check for shadows or blur.")
    print("3. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for easier coordination
        frame = cv2.flip(frame, 1)
        
        # Create a display copy to draw the UI
        # (This keeps the raw frame 'clean' for the ROI preview)
        display_frame = frame.copy()
        
        # Draw the ROI Bounding Box (White)
        cv2.rectangle(display_frame, (X1, Y1), (X2, Y2), (255, 255, 255), 2)
        
        # Crop the ROI to show what the model will actually 'see'
        roi_preview = frame[Y1:Y2, X1:X2]
        # Resize the preview so it's easy to see in a separate window
        roi_preview = cv2.resize(roi_preview, (300, 300))

        # Labels
        cv2.putText(display_frame, "MAIN FEED", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show both the full feed and the cropped version
        cv2.imshow("1. Full Feed (Positioning)", display_frame)
        cv2.imshow("2. ROI Preview (What is saved)", roi_preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()