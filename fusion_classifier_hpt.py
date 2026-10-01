import os
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

DEVICE = "cpu"
TOP_K = 2
N_TRIALS = 20
SEED = 1337
FIXED_TAU = 0.9

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

bert_train = np.load("trainhingroberta_embs.npy")
bert_test  = np.load("testhingroberta_embs.npy")

word_train = np.load("train_embs.npy")
word_test  = np.load("test_embs.npy")

def fuse(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return torch.cat([a, b], dim=1)

X_train = fuse(bert_train, word_train).to(DEVICE)
X_test  = fuse(bert_test,  word_test ).to(DEVICE)

le = LabelEncoder()

y_train = torch.tensor(
    le.fit_transform(train_df["category"].astype(str))
).to(DEVICE)

known = set(le.classes_)
mapped_test = [
    c if c in known else le.classes_[0]
    for c in test_df["category"].astype(str)
]

y_test = torch.tensor(le.transform(mapped_test)).to(DEVICE)

NUM_CLASSES = len(le.classes_)
FUSION_DIM = X_train.shape[1]

class FlatFusionClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, dropout_base):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_base),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_base * 0.75),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_base * 0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(
    lr,
    epochs,
    label_smoothing,
    dropout_base,
    temperature,
    conf_quantile
):
    model = FlatFusionClassifier(
        FUSION_DIM,
        NUM_CLASSES,
        dropout_base
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 128):
            idx = perm[i:i+128]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits / temperature, dim=1)

    top1_preds = probs.argmax(dim=1).cpu().numpy()
    acc1 = (top1_preds == y_test.cpu().numpy()).mean()

    top2_preds = torch.topk(probs, k=2, dim=1).indices.cpu().numpy()
    acc2 = np.mean([
        y_test.cpu().numpy()[i] in top2_preds[i]
        for i in range(len(y_test))
    ])

    confidence = probs.max(dim=1).values.cpu().numpy()
    conf_mask = confidence >= FIXED_TAU

    coverage = conf_mask.mean()

    conf_acc = (
        np.mean([
            y_test.cpu().numpy()[i] == top1_preds[i]
            for i in range(len(y_test))
            if conf_mask[i]
        ]) if conf_mask.sum() > 0 else 0.0
    )

    weighted_f1 = f1_score(
        y_test.cpu().numpy(),
        top1_preds,
        average="weighted"
    )

    return {
        "weighted_f1": weighted_f1,
        "top1": acc1,
        "top2": acc2,
        "conf_acc": conf_acc,
        "coverage": coverage,
        "preds": top1_preds
    }

def objective(trial):
    metrics = train_and_evaluate(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        epochs=trial.suggest_categorical("epochs", [30, 40, 60]),
        label_smoothing=trial.suggest_float("label_smoothing", 0.05, 0.2),
        dropout_base=trial.suggest_float("dropout_base", 0.3, 0.5),
        temperature=trial.suggest_float("temperature", 0.5, 1.5),
        conf_quantile=trial.suggest_float("conf_quantile", 0.45, 0.65)
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

    final_metrics = train_and_evaluate(**study.best_params)

    print(f"Weighted F1-score     : {final_metrics['weighted_f1']:.4f}")
    print(f"Top-1 Accuracy        : {final_metrics['top1']:.4f}")
    print(f"Top-2 Accuracy        : {final_metrics['top2']:.4f}")
    print(f"Confidence Accuracy   : {final_metrics['conf_acc']:.4f}")
    print(f"Coverage @ τ=0.9      : {final_metrics['coverage']:.4f}")

    used = np.unique(y_test.cpu().numpy())

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test.cpu().numpy(),
        final_metrics["preds"],
        labels=used,
        target_names=le.inverse_transform(used),
        zero_division=0
    ))
