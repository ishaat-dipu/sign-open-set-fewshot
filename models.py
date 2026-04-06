import torch
import torch.nn as nn

class LandmarkMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.net(x)

class EmbedNet(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
    def forward(self, x):
        z = self.body(x)
        return z / (z.norm(dim=1, keepdim=True) + 1e-8)  # L2 Normalize for Open-Set

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.head(self.conv(x))

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())