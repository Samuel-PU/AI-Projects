#!/usr/bin/env python3
# League of Legends Match Predictor — Improved Pipeline (Steps 1–8)

import argparse, os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

import matplotlib.pyplot as plt


# ----------------------------
# Model (logits out; NO sigmoid here)
# ----------------------------
class LogisticRegressionModel(nn.Module):
    """Binary logistic regression with logits output."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # logits


# ----------------------------
# Utilities
# ----------------------------
def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return preds.eq(y_true).float().mean().item()

def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close()

def make_loaders(X_train_t, y_train_t, batch_size=64, shuffle=True):
    ds = TensorDataset(X_train_t, y_train_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def compute_pos_weight(y_train_t: torch.Tensor) -> torch.Tensor:
    pos = y_train_t.sum().item()
    neg = y_train_t.numel() - pos
    ratio = (neg / max(pos, 1e-8)) if pos > 0 else 1.0
    return torch.tensor([ratio], dtype=torch.float32)


# ----------------------------
# Step 1: Load, split, scale, tensors (+ optional interactions)
# ----------------------------
def step1_load_preprocess(csv_path: str, target_col: str = "win",
                          test_size: float = 0.2, seed: int = 42,
                          use_interactions: bool = False):
    print("== Step 1: Loading and preprocessing data ==")
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    X_df = df.drop(columns=[target_col])
    y_s = df[target_col].astype(np.float32)

    # Train/test split (stratify if binary)
    strat = y_s if set(y_s.unique()) <= {0.0, 1.0} else None
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X_df, y_s, test_size=test_size, random_state=seed, stratify=strat
    )

    # Optional interactions (pairwise, no squares), then standardise
    if use_interactions:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_np = poly.fit_transform(X_train_df.values)
        X_test_np  = poly.transform(X_test_df.values)
        feature_names = poly.get_feature_names_out(X_train_df.columns)
    else:
        X_train_np = X_train_df.values
        X_test_np  = X_test_df.values
        feature_names = np.array(X_train_df.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_np)
    X_test  = scaler.transform(X_test_np)

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s.values.reshape(-1, 1), dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_s.values.reshape(-1, 1), dtype=torch.float32)

    print(f"Train shape: {X_train_t.shape}  {y_train_t.shape}")
    print(f"Test  shape: {X_test_t.shape}  {y_test_t.shape}")
    return (X_train_df, X_test_df, feature_names, X_train_t, X_test_t, y_train_t, y_test_t)


# ----------------------------
# Step 3: Train loop (mini-batch)
# ----------------------------
def step3_train_minibatch(model, optimizer, criterion, X_train_t, y_train_t,
                          epochs: int = 1000, batch_size: int = 64, print_every: int = 100):
    print("\n== Step 3: Training (mini-batch; BCEWithLogitsLoss) ==")
    loader = make_loaders(X_train_t, y_train_t, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        if epoch % print_every == 0:
            avg_loss = running_loss / len(loader.dataset)
            with torch.no_grad():
                logits_full = model(X_train_t)
                train_acc = accuracy_from_logits(logits_full, y_train_t, threshold=0.5)
            print(f"Epoch {epoch:04d} | loss={avg_loss:.6f} | train_acc={train_acc:.4f}")


def evaluate_model(model, X_train_t, y_train_t, X_test_t, y_test_t, label: str = "", threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t)
        test_logits  = model(X_test_t)
        train_acc = accuracy_from_logits(train_logits, y_train_t, threshold)
        test_acc  = accuracy_from_logits(test_logits,  y_test_t,  threshold)
    print(f"{label} Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")
    return train_logits, test_logits, train_acc, test_acc


# ----------------------------
# Step 4: Retrain with L2 (weight decay)
# ----------------------------
def step4_train_l2(input_dim, pos_weight, X_train_t, y_train_t, X_test_t, y_test_t,
                   lr=0.01, weight_decay=0.01, epochs=1000, batch_size=64):
    print("\n== Step 4: Training with L2 (weight_decay) ==")
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    step3_train_minibatch(model, optimizer, criterion, X_train_t, y_train_t, epochs=epochs, batch_size=batch_size)
    _, _, tr_acc, te_acc = evaluate_model(model, X_train_t, y_train_t, X_test_t, y_test_t, "[L2]")
    return model, tr_acc, te_acc


# ----------------------------
# Step 5: Visualisation & reports (+ Youden threshold)
# ----------------------------
def step5_visualise_and_report(model, X_test_t, y_test_t, outdir="outputs"):
    print("\n== Step 5: Visualisation & reports (ROC/CM/Report + Youden threshold) ==")
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t).cpu().numpy().ravel()
        test_probs  = 1 / (1 + np.exp(-test_logits))  # sigmoid
        y_true      = y_test_t.cpu().numpy().ravel()

    # ROC & AUC
    fpr, tpr, thr = roc_curve(y_true, test_probs)
    auc_val = roc_auc_score(y_true, test_probs)
    print(f"AUC: {auc_val:.4f}")

    # Youden J threshold
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(thr[best_idx])
    print(f"Best threshold by Youden J: {best_thr:.4f}")

    # Confusion Matrix & Report at best threshold
    preds = (test_probs >= best_thr).astype(int)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, digits=4)
    print("Confusion Matrix @ best_thr:\n", cm)
    print("\nClassification Report @ best_thr:\n", report)

    # Save plots & report
    # Confusion Matrix plot
    plt.figure(figsize=(4.8, 4.8))
    im = plt.imshow(cm, interpolation="nearest")
    plt.colorbar(im)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix @ best_thr")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    save_fig(os.path.join(outdir, "confusion_matrix_best_thr.png"))

    # ROC plot
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.grid(True, alpha=0.3)
    save_fig(os.path.join(outdir, "roc_curve.png"))

    # Save text report
    rep_path = os.path.join(outdir, "classification_report_best_thr.txt")
    with open(rep_path, "w") as f:
        f.write(f"AUC: {auc_val:.6f}\n")
        f.write(f"Best threshold (Youden J): {best_thr:.6f}\n\n")
        f.write("Confusion Matrix @ best_thr:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Classification Report @ best_thr:\n")
        f.write(report + "\n")
    print(f"[Saved] {rep_path}")

    return best_thr


# ----------------------------
# Step 6: Save & load
# ----------------------------
def step6_save_load(model, input_dim, X_test_t, y_test_t, path="outputs/lol_logreg_l2.pth", threshold=0.5):
    print("\n== Step 6: Save & Load ==")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Saved model] {path}")

    loaded = LogisticRegressionModel(input_dim)
    loaded.load_state_dict(torch.load(path, map_location="cpu"))
    loaded.eval()
    with torch.no_grad():
        logits = loaded(X_test_t)
        acc = accuracy_from_logits(logits, y_test_t, threshold=threshold)
    print(f"Loaded model test acc @ thr={threshold:.3f}: {acc:.4f}")
    return acc


# ----------------------------
# Step 7: Hyperparameter tuning grid (lr × weight_decay)
# ----------------------------
def step7_tune_lr_wd(input_dim, pos_weight, X_train_t, y_train_t, X_test_t, y_test_t,
                     lrs=(0.01, 0.05, 0.1), wds=(0.0, 1e-4, 1e-3, 1e-2),
                     epochs=300, batch_size=64):
    print("\n== Step 7: Hyperparameter tuning (learning rate × weight_decay) ==")
    results = {}
    for lr in lrs:
        for wd in wds:
            m = LogisticRegressionModel(input_dim)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            opt = optim.SGD(m.parameters(), lr=lr, weight_decay=wd)
            # mini-batch train
            loader = make_loaders(X_train_t, y_train_t, batch_size=batch_size, shuffle=True)
            m.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    logits = m(xb)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
            # eval @ 0.5
            m.eval()
            with torch.no_grad():
                te_logits = m(X_test_t)
                te_acc = accuracy_from_logits(te_logits, y_test_t, threshold=0.5)
            results[(lr, wd)] = te_acc
            print(f"lr={lr:.3f}, wd={wd:.0e} -> test acc={te_acc:.4f}")
    best = max(results, key=results.get)
    print(f"Best: lr={best[0]}, wd={best[1]}  (test acc={results[best]:.4f})")
    return best, results


# ----------------------------
# Step 8: Feature importance (weights)
# ----------------------------
def step8_feature_importance(model, feature_names, outdir="outputs"):
    print("\n== Step 8: Feature importance ==")
    os.makedirs(outdir, exist_ok=True)
    weights = model.linear.weight.detach().cpu().numpy().ravel()
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "weight": weights,
        "abs_weight": np.abs(weights),
    }).sort_values("abs_weight", ascending=False).reset_index(drop=True)

    print("Top 10 by |weight|:")
    print(fi_df.head(10))

    # Plot top 20
    topk = min(20, len(fi_df))
    plot_df = fi_df.head(topk)[::-1]
    plt.figure(figsize=(9, 0.45 * topk + 2))
    plt.barh(plot_df["feature"], plot_df["weight"])
    plt.xlabel("Weight (importance)")
    plt.title("Logistic Regression Feature Importance")
    save_fig(os.path.join(outdir, "feature_importance.png"))

    csv_path = os.path.join(outdir, "feature_importance.csv")
    fi_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV with binary 'win' column")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--use_interactions", action="store_true", help="Add pairwise interaction features")
    args = ap.parse_args()

    # Step 1
    (X_train_df, X_test_df, feature_names,
     X_train_t, X_test_t, y_train_t, y_test_t) = step1_load_preprocess(
        args.csv, target_col="win", seed=args.seed, use_interactions=args.use_interactions
    )

    # Step 2
    print("\n== Step 2: Model, loss, optimiser init ==")
    input_dim = X_train_t.shape[1]
    pos_weight = compute_pos_weight(y_train_t)
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Initialised. lr={args.lr}, weight_decay={args.weight_decay}, pos_weight={pos_weight.item():.4f}")

    # Step 3
    step3_train_minibatch(model, optimizer, criterion, X_train_t, y_train_t,
                          epochs=args.epochs, batch_size=args.batch_size)
    train_logits, test_logits, tr_acc, te_acc = evaluate_model(
        model, X_train_t, y_train_t, X_test_t, y_test_t, "[No threshold tuning]", threshold=0.5
    )

    # Step 4 (re-train with L2 already covered by weight_decay; keep explicit demo)
    model_l2, tr_acc_l2, te_acc_l2 = step4_train_l2(
        input_dim, pos_weight, X_train_t, y_train_t, X_test_t, y_test_t,
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.batch_size
    )

    # Step 5 (Youden threshold)
    best_thr = step5_visualise_and_report(model_l2, X_test_t, y_test_t, outdir=args.outdir)

    # Step 6 (save/load with tuned threshold)
    save_path = os.path.join(args.outdir, "lol_logreg_l2.pth")
    loaded_acc = step6_save_load(model_l2, input_dim, X_test_t, y_test_t, path=save_path, threshold=best_thr)

    # Step 7 (grid tuning)
    best_combo, grid_results = step7_tune_lr_wd(
        input_dim, pos_weight, X_train_t, y_train_t, X_test_t, y_test_t,
        lrs=(0.01, 0.05, 0.1), wds=(0.0, 1e-4, 1e-3, 1e-2),
        epochs=300, batch_size=args.batch_size
    )

    # Step 8 (feature importance)
    step8_feature_importance(model_l2, feature_names, outdir=args.outdir)

    # Summary
    print("\n== Summary ==")
    print(f"[Initial] Train acc: {tr_acc:.4f} | Test acc: {te_acc:.4f} (thr=0.5)")
    print(f"[L2     ] Train acc: {tr_acc_l2:.4f} | Test acc: {te_acc_l2:.4f} (thr=0.5)")
    print(f"[Loaded ] Test acc:  {loaded_acc:.4f} (thr={best_thr:.4f})")
    print(f"[Best grid] lr={best_combo[0]}, wd={best_combo[1]}  (test acc={grid_results[best_combo]:.4f})")
    print(f"Outputs saved under: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
