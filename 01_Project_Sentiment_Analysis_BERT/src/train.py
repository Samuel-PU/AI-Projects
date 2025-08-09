import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.calibration import calibration_curve
from pathlib import Path
import os
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt

# ==== Config ====
MODEL_DIR = Path("../models/bert_sentiment")
PLOTS_DIR = Path("./plots")
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 5e-5
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KFOLD = True  # set False to disable cross-validation
N_FOLDS = 5
MIN_SAMPLES_FOR_KFOLD = 50  # require at least this many examples to do k-fold

# ==== Paths ====
current_dir = Path(__file__).parent
data_path = current_dir.parent / "data" / "sample_reviews_expanded.csv"  # updated larger dataset

# ==== Data creation fallback ====
if not data_path.exists():
    print("⚠️ Expanded data not found; falling back to creating tiny sample.")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data = '''text,label
"This movie was amazing!",1
"Terrible acting and plot.",0
"I loved the character development.",1
"Worst film of the year.",0
"The cinematography was stunning.",1
'''
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(sample_data)

# ==== Load ====
df = pd.read_csv(data_path)
print("✅ Loaded data, sample:")
print(df.head())

# ==== Prepare inputs ====
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

# ==== Tokenizer & Dataset ====
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

full_dataset = SentimentDataset(texts, labels, tokenizer, MAX_LENGTH)

# ==== Utility training/eval function ====
def train_and_evaluate(train_idx, val_idx, fold_id=None):
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    loss_history = []
    val_acc_history = []
    all_val_preds = []
    all_val_labels = []

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Fold {fold_id or 0} Train Epoch {epoch+1}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Validation pass
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labels_batch = batch["labels"]
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)
                all_val_preds.extend(preds.cpu().tolist())
                all_val_labels.extend(labels_batch.cpu().tolist())
        accuracy = correct / total if total > 0 else 0
        val_acc_history.append(accuracy)
        print(f"[Fold {fold_id}] Epoch {epoch+1} train loss {avg_loss:.4f} val acc {accuracy:.2f}")
        model.train()

    # Final eval metrics
    report = classification_report(all_val_labels, all_val_preds, target_names=["neg", "pos"], zero_division=0, output_dict=True)
    cm_raw = confusion_matrix(all_val_labels, all_val_preds, labels=[0,1])
    cm_norm = confusion_matrix(all_val_labels, all_val_preds, labels=[0,1], normalize="true")

    # Calibration (reliability) diagram data
    model.eval()
    probs = []
    true = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            softmax = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pos_prob = softmax[:,1]
            probs.extend(pos_prob.cpu().tolist())
            true.extend(batch["labels"].cpu().tolist())
    fraction_of_positives, mean_predicted_value = calibration_curve(true, probs, n_bins=10, strategy="uniform")

    # Save per-fold plots
    suffix = f"_fold{fold_id}" if fold_id is not None else ""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss & accuracy curves
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss per Epoch{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"training_loss{suffix}.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), val_acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy per Epoch{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"validation_accuracy{suffix}.png")
    plt.close()

    # Confusion matrices
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=["neg", "pos"])
    plt.figure()
    disp_raw.plot(ax=plt.gca(), cmap=None, colorbar=False)
    plt.title(f"Confusion Matrix (raw){suffix}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_raw{suffix}.png")
    plt.close()

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["neg", "pos"])
    plt.figure()
    disp_norm.plot(ax=plt.gca(), cmap=None, colorbar=False)
    plt.title(f"Confusion Matrix (normalized){suffix}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_norm{suffix}.png")
    plt.close()

    # Reliability diagram
    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, marker="o")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability Diagram{suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"reliability_diagram{suffix}.png")
    plt.close()

    return {
        "loss_history": loss_history,
        "val_acc_history": val_acc_history,
        "classification_report": report,
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "calibration_curve": (mean_predicted_value, fraction_of_positives),
        "preds": all_val_preds,
        "labels": all_val_labels,
        "model": model,  # last fold's model
        "tokenizer": tokenizer
    }

# ==== Execution ====
results = []
if KFOLD and len(full_dataset) >= MIN_SAMPLES_FOR_KFOLD:
    print(f"Running {N_FOLDS}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    X = texts  # not used directly
    y = labels
    for fold_id, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        # need corresponding train indices; skf.split gives train/test pattern, so recompute
        train_idx = [i for i in range(len(y)) if i not in val_idx]
        r = train_and_evaluate(train_idx, val_idx, fold_id=fold_id)
        results.append(r)
else:
    # single split with fallback stratify logic
    print("Running single train/validation split...")
    from collections import Counter
    n_classes = len(set(labels))
    n_samples = len(labels)
    desired_frac = 0.2
    test_size_count = max(1, int(n_samples * desired_frac))
    if test_size_count < n_classes:
        test_size = n_classes / n_samples
        stratify_arg = None
    else:
        test_size = desired_frac
        stratify_arg = labels
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=stratify_arg
    )

    # get indices
    label_list = labels  # original
    # build index lists by matching
    train_idx = []
    val_idx = []
    used_val = set()
    for i, (t, l) in enumerate(zip(texts, labels)):
        if len(val_idx) < len(val_labels) and l in val_labels and (i not in used_val):
            val_idx.append(i)
            used_val.add(i)
        else:
            train_idx.append(i)
    r = train_and_evaluate(train_idx, val_idx, fold_id=None)
    results.append(r)

# ==== Save final model/tokenizer from last result ====
best = results[-1]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
best["model"].save_pretrained(MODEL_DIR)
best["tokenizer"].save_pretrained(MODEL_DIR)
print(f"✅ Saved model/tokenizer to {MODEL_DIR.resolve()}")

# ==== Aggregate and print summary ====
def summarize_report(rep):
    print("Classification report (averaged):")
    df_rep = pd.DataFrame(rep["classification_report"]).transpose()
    print(df_rep.loc[["neg", "pos", "accuracy", "macro avg", "weighted avg"]])

if len(results) > 1:
    # average metrics across folds
    accs = [r["val_acc_history"][-1] for r in results]
    print(f"Validation accuracies across folds: {accs}")
    # print first fold report as representative
    summarize_report(results[0])
else:
    summarize_report(results[0])
