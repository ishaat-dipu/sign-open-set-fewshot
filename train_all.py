import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import LandmarkMLP, EmbedNet, TinyCNN

def roi_to_tensor(X_roi_uint8):
    return torch.from_numpy(X_roi_uint8).permute(0, 3, 1, 2).float() / 255.0

def train_model(model, dl_tr, device, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
    return model.state_dict()

def main():
    ds = np.load("artifacts/dataset.npz", allow_pickle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, D = len(ds["classes"]), ds["Xtr_lm"].shape[1]

    print(f"Training on {device} for {C} classes...")

    # 1. Landmark MLP
    lm_model = LandmarkMLP(D, C).to(device)
    dl_lm = DataLoader(TensorDataset(torch.from_numpy(ds["Xtr_lm"]), torch.from_numpy(ds["ytr"])), batch_size=64, shuffle=True)
    torch.save({"in_dim": D, "state_dict": train_model(lm_model, dl_lm, device)}, "artifacts/landmark_mlp.pt")
    print(" Landmark-MLP Trained")

    # 2. EmbedNet (Proposed)
    emb_dim = 32
    emb_model = EmbedNet(D, emb_dim).to(device)
    head = nn.Linear(emb_dim, C).to(device)
    opt = torch.optim.Adam(list(emb_model.parameters()) + list(head.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    emb_model.train(); head.train()
    for ep in range(30):
        for xb, yb in dl_lm:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(head(emb_model(xb)), yb)
            loss.backward()
            opt.step()
    torch.save({"in_dim": D, "emb_dim": emb_dim, "emb": emb_model.state_dict()}, "artifacts/embednet.pt")
    print(" EmbedNet Trained")

    # 3. TinyCNN
    cnn_model = TinyCNN(C).to(device)
    dl_cnn = DataLoader(TensorDataset(roi_to_tensor(ds["Xtr_roi"]), torch.from_numpy(ds["ytr"])), batch_size=32, shuffle=True)
    torch.save({"state_dict": train_model(cnn_model, dl_cnn, device, epochs=15)}, "artifacts/tiny_cnn.pt")
    print(" TinyCNN Trained")

if __name__ == "__main__":
    main()