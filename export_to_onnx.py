import os
import torch
import numpy as np
from models import LandmarkMLP, EmbedNet, TinyCNN

def export_model(model, dummy_input, save_path):
    model.eval()
    print(f"Exporting to {save_path}...")
    
    # We remove the deprecated opset and dynamo warnings for a clean export
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Successfully exported {save_path}\n")

def main():
    os.makedirs("artifacts", exist_ok=True)
    device = "cpu"
    
    # Load dataset to get the correct number of classes
    ds = np.load("artifacts/dataset.npz", allow_pickle=True)
    C = len(ds["classes"])
    
    # 1. Export Proposed EmbedNet
    if os.path.exists("artifacts/embednet.pt"):
        ck = torch.load("artifacts/embednet.pt", map_location=device)
        emb_model = EmbedNet(in_dim=ck["in_dim"], emb_dim=ck["emb_dim"])
        emb_model.load_state_dict(ck["emb"])
        dummy_input = torch.randn(1, ck["in_dim"])
        export_model(emb_model, dummy_input, "artifacts/embednet.onnx")

    # 2. Export Landmark-MLP
    if os.path.exists("artifacts/landmark_mlp.pt"):
        ck = torch.load("artifacts/landmark_mlp.pt", map_location=device)
        lm_model = LandmarkMLP(in_dim=ck["in_dim"], num_classes=C)
        lm_model.load_state_dict(ck["state_dict"])
        dummy_input = torch.randn(1, ck["in_dim"])
        export_model(lm_model, dummy_input, "artifacts/landmark_mlp.onnx")

    # 3. Export TinyCNN
    if os.path.exists("artifacts/tiny_cnn.pt"):
        ck = torch.load("artifacts/tiny_cnn.pt", map_location=device)
        cnn_model = TinyCNN(num_classes=C)
        cnn_model.load_state_dict(ck["state_dict"])
        dummy_input = torch.randn(1, 3, 96, 96) # Image input: Batch, Channels, H, W
        export_model(cnn_model, dummy_input, "artifacts/tiny_cnn.onnx")

if __name__ == "__main__":
    main()