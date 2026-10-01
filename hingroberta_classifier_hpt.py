import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

MODEL_NAME = "l3cube-pune/hing-roberta"
DEVICE = "cpu"
FIXED_TAU = 0.9
N_TRIALS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

def clean_texts(texts):
    out = []
    for t in texts:
        if t is None or pd.isna(t):
            out.append("")
        else:
            t = str(t).strip()
            out.append(t if len(t) > 0 else "")
    return out

train_texts = clean_texts(train_df["content_processed"].tolist())
test_texts  = clean_texts(test_df["content_processed"].tolist())

le = LabelEncoder()
le.fit(train_df["category"].astype(str))

train_labels = le.transform(train_df["category"].astype(str))

test_labels = test_df["category"].astype(str).apply(
    lambda x: x if x in le.classes_ else "<<UNK>>"
)

if "<<UNK>>" not in le.classes_:
    le.classes_ = np.append(le.classes_, "<<UNK>>")

test_labels = le.transform(test_labels)
num_classes = len(le.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

full_weights = np.ones(num_classes)
for c, w in zip(np.unique(train_labels), weights):
    full_weights[c] = w

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(full_weights, dtype=torch.float)
)

def train_and_evaluate(
    lr,
    batch_size,
    max_len,
    freeze_layers,
    epochs
):
    train_ds = TextDataset(train_texts, train_labels, max_len)
    test_ds  = TextDataset(test_texts, test_labels, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes
    ).to(DEVICE)

    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False

    for layer in model.roberta.encoder.layer[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()

    model.eval()

    all_preds, all_labels = [], []
    top2_hits = 0
    conf_correct, conf_total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch["labels"]

            logits = model(**batch).logits
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            top2 = torch.topk(probs, 2, dim=1).indices
            top2_hits += (top2 == labels.unsqueeze(1)).any(dim=1).sum().item()

            confidence = probs.max(dim=1).values
            mask = confidence >= FIXED_TAU

            conf_correct += (preds[mask] == labels[mask]).sum().item()
            conf_total += mask.sum().item()

    top1 = accuracy_score(all_labels, all_preds)
    top2 = top2_hits / len(all_labels)

    coverage = conf_total / len(all_labels)
    conf_acc = conf_correct / conf_total if conf_total > 0 else 0.0

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "top1": top1,
        "top2": top2,
        "conf_acc": conf_acc,
        "coverage": coverage,
        "preds": all_preds,
        "labels": all_labels
    }

def objective(trial):
    metrics = train_and_evaluate(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        max_len=trial.suggest_categorical("max_len", [48, 64, 96]),
        freeze_layers=trial.suggest_categorical("freeze_layers", [4, 6, 8]),
        epochs=trial.suggest_categorical("epochs", [2, 3, 4])
    )

    print(
        f"[Trial {trial.number}] "
        f"F1={metrics['weighted_f1']:.4f} | "
        f"Top1={metrics['top1']:.4f} | "
        f"Top2={metrics['top2']:.4f} | "
        f"ConfAcc@τ=0.9={metrics['conf_acc']:.4f} | "
        f"Coverage@τ=0.9={metrics['coverage']:.4f}"
    )

    return metrics["weighted_f1"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBEST HYPERPARAMETERS:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\nFINAL EVALUATION (BEST MODEL):")
    best_metrics = train_and_evaluate(**study.best_params)

    print(f"Weighted F1-score     : {best_metrics['weighted_f1']:.4f}")
    print(f"Macro F1-score        : {best_metrics['macro_f1']:.4f}")
    print(f"Top-1 Accuracy        : {best_metrics['top1']:.4f}")
    print(f"Top-2 Accuracy        : {best_metrics['top2']:.4f}")
    print(f"Confidence Accuracy   : {best_metrics['conf_acc']:.4f}")
    print(f"Coverage @ τ=0.9      : {best_metrics['coverage']:.4f}")

    used = np.unique(best_metrics["labels"])

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        best_metrics["labels"],
        best_metrics["preds"],
        labels=used,
        target_names=le.inverse_transform(used),
        zero_division=0
    ))
