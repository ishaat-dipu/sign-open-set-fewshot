import onnx

def main():
    print("Loading split model...")
    # This automatically loads the .onnx and the .data file together
    model = onnx.load("artifacts/landmark_mlp.onnx")
    
    print("Saving as a single unified file...")
    # This forces ONNX to pack everything into one single file
    onnx.save_model(model, "artifacts/landmark_mlp_web.onnx")
    
    print(" Fixed! Model is now a single file ready for the browser.")

if __name__ == "__main__":
    main()