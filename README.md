PROTOSIGN: A ZERO-RETRAINING PROTOTYPICAL NETWORK FOR EDGE-BASED SIGN LANGUAGE RECOGNITION

OVERVIEW
ProtoSign is an ultra-lightweight Edge AI system designed for real-time Sign Language Recognition. By combining MediaPipe's geometric landmark extraction with a custom Prototypical EmbedNet, this system achieves 99.04 percent accuracy using only 20,000 parameters. 

Unlike traditional classification models, ProtoSign operates in a discriminative metric space. This enables Few-Shot Learning (instantly learning new signs without retraining) and Open-Set Rejection (actively rejecting unknown gestures).

KEY FEATURES
- Zero-Retraining Dynamic Extension: Teach the model a new sign in real-time simply by showing it to the camera and typing a name. No backpropagation required.
- Open-Set Rejection: Mathematically measures gesture familiarity. If a sign is too far from known prototypes, it actively rejects it as UNKNOWN.
- Edge Optimized: 60 times smaller than standard TinyCNN baselines. Inference latency is 0.0134 ms, making it ideal for low-power IoT, mobile NPUs, and WebAssembly deployment.
- Environmentally Robust: Immune to motion blur, lighting changes, and skin-tone bias due to scale-normalized skeletal feature extraction.

INSTALLATION AND SETUP
1. Clone the repository to your local machine.
2. Create and activate a Python virtual environment.
3. Install the required dependencies (OpenCV, PyTorch, NumPy, MediaPipe).
4. Download the MediaPipe hand_landmarker.task model and place it in the models directory.

USAGE: RUNNING THE LIVE SYSTEM
Run the live_camera.py script to start the real-time recognition system using your webcam.

How to use Teach Mode (Few-Shot Learning):
1. Hold up a gesture the system does not know. The UI box will display UNKNOWN.
2. While keeping your hand in frame, press the N key on your keyboard.
3. The video will pause. Look at your terminal and type the name of the new sign (e.g., VICTORY).
4. Press Enter. The system will instantly average the 32-dimensional embedding, save the new Prototype, and recognize your new sign.
5. Press Q to quit the application.

SYSTEM ARCHITECTURE
The ProtoSign pipeline consists of two distinct phases:

1. Skeletal Feature Extraction: Raw RGB pixels are discarded. MediaPipe extracts 21 3D hand joints. These are translation-centered to the wrist and scale-normalized, creating a robust 83-dimensional feature vector (63 joint coordinates and 20 bone distances).

2. Metric Learning (EmbedNet): Instead of using a closed-set Softmax layer, our lightweight 3-layer MLP compresses the 83-dimensional geometry into a highly optimized 32-dimensional latent vector. Classification is performed by measuring the Cosine Similarity to saved mathematical Prototypes.

PERFORMANCE BENCHMARK
TinyCNN (Baseline): 92.4 percent accuracy, 1.2 Million parameters, 5.2 ms latency, fails under motion blur.
Landmark-MLP: 98.1 percent accuracy, 20,412 parameters, 0.01 ms latency, excellent blur robustness.
Proposed EmbedNet: 99.04 percent accuracy, 20,412 parameters, 0.0134 ms latency, excellent blur robustness.

RESEARCH PAPER
This repository contains the source code for our research paper: "ProtoSign: A Zero-Retraining Prototypical Network for Edge-Based Sign Language Recognition" (Authors: Ishaat Rahman Dipu, Subrata K. Paul). Read the full methodology and results in the included PDF or documentation folders.
