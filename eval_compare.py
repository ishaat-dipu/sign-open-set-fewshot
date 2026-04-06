import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models import LandmarkMLP, EmbedNet, TinyCNN, count_params

def latency_ms(model, x, device):
    xb = x.to(device)
    with torch.no_grad():
        for _ in range(10): model(xb) # Warmup
        t0 = time.time()
        for _ in range(500): model(xb)
    return ((time.time() - t0) / 500) * 1000

def main():
    ds = np.load("artifacts/dataset.npz", allow_pickle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C, D = len(ds["classes"]), ds["Xte_lm"].shape[1]

    Xte_lm = torch.from_numpy(ds["Xte_lm"]).to(device)
    yte = ds["yte"]
    Xte_roi = (torch.from_numpy(ds["Xte_roi"]).permute(0,3,1,2).float()/255.0).to(device)

    results = []

    # 1. Landmark MLP Eval
    ck_lm = torch.load("artifacts/landmark_mlp.pt", map_location="cpu")
    lm = LandmarkMLP(D, C).to(device)
    lm.load_state_dict(ck_lm["state_dict"])
    lm.eval()
    with torch.no_grad(): preds_lm = torch.argmax(lm(Xte_lm), dim=1).cpu().numpy()
    results.append(("Landmark-MLP", accuracy_score(yte, preds_lm), f1_score(yte, preds_lm, average="macro"), latency_ms(lm, Xte_lm[:1], device), count_params(lm)))

    # 2. TinyCNN Eval
    ck_cnn = torch.load("artifacts/tiny_cnn.pt", map_location="cpu")
    cnn = TinyCNN(C).to(device)
    cnn.load_state_dict(ck_cnn["state_dict"])
    cnn.eval()
    with torch.no_grad(): preds_cnn = torch.argmax(cnn(Xte_roi), dim=1).cpu().numpy()
    results.append(("TinyCNN", accuracy_score(yte, preds_cnn), f1_score(yte, preds_cnn, average="macro"), latency_ms(cnn, Xte_roi[:1], device), count_params(cnn)))

    # 3. Proposed EmbedNet Eval
    ck_emb = torch.load("artifacts/embednet.pt", map_location="cpu")
    emb = EmbedNet(D, ck_emb["emb_dim"]).to(device)
    emb.load_state_dict(ck_emb["emb"])
    emb.eval()
    with torch.no_grad():
        Ztr = emb(torch.from_numpy(ds["Xtr_lm"]).to(device)).cpu().numpy()
        Zte = emb(Xte_lm).cpu().numpy()
    
    protos = np.stack([Ztr[ds["ytr"] == c].mean(axis=0) for c in range(C)])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    
    sims = Zte @ protos.T
    preds_prop = np.argmax(sims, axis=1)
    results.append(("Proposed(Proto)", accuracy_score(yte, preds_prop), f1_score(yte, preds_prop, average="macro"), latency_ms(emb, Xte_lm[:1], device), count_params(emb)))

    print("\n=== Closed-set Comparison ===")
    print(f"{'Method':<18} | {'Acc':>8} | {'MacroF1':>8} | {'Lat(ms)':>8} | {'Params':>8}")
    print("-" * 55)
    for r in results: print(f"{r[0]:<18} | {r[1]:8.4f} | {r[2]:8.4f} | {r[3]:8.3f} | {r[4]:8d}")

    # Open-Set Eval
    if "Xunk_lm" in ds.files:
        print("\n=== Open-Set AUROC (Known vs Unknown) ===")
        Xunk_lm = torch.from_numpy(ds["Xunk_lm"]).to(device)
        with torch.no_grad():
            score_k_lm = torch.softmax(lm(Xte_lm), dim=1).max(dim=1).values.cpu().numpy()
            score_u_lm = torch.softmax(lm(Xunk_lm), dim=1).max(dim=1).values.cpu().numpy()
            print(f"Landmark-MLP: {roc_auc_score(np.concatenate([np.ones_like(score_k_lm), np.zeros_like(score_u_lm)]), np.concatenate([score_k_lm, score_u_lm])):.4f}")

            sim_unk = emb(Xunk_lm).cpu().numpy() @ protos.T
            print(f"Proposed:     {roc_auc_score(np.concatenate([np.ones_like(sims.max(axis=1)), np.zeros_like(sim_unk.max(axis=1))]), np.concatenate([sims.max(axis=1), sim_unk.max(axis=1)])):.4f}")

if __name__ == "__main__":
    main()