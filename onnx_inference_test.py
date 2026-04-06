import onnxruntime as ort
import numpy as np
import time

def main():
    # 1. Load the ONNX Runtime Session (CPU Provider)
    # This is lightning fast and doesn't require massive PyTorch libraries
    session = ort.InferenceSession("artifacts/embednet.onnx", providers=['CPUExecutionProvider'])
    
    # Get the name of the input layer that we defined during export
    input_name = session.get_inputs()[0].name
    
    # Create fake feature data to simulate MediaPipe (1 sample, 83 features)
    dummy_features = np.random.randn(1, 83).astype(np.float32)

    # 2. Warmup run (first run is always slightly slower due to memory allocation)
    for _ in range(10):
        session.run(None, {input_name: dummy_features})

    # 3. Benchmark speed
    start = time.time()
    for _ in range(1000):
        # This is how you predict!
        output = session.run(None, {input_name: dummy_features})[0]
    end = time.time()

    latency = ((end - start) / 1000) * 1000
    print(f"ONNX Output Shape: {output.shape} (Batch, 32D Embedding)")
    print(f"ONNX CPU Latency: {latency:.4f} ms per frame")
    print(f"ONNX Max FPS: {1000 / latency:.0f}")

if __name__ == "__main__":
    main()