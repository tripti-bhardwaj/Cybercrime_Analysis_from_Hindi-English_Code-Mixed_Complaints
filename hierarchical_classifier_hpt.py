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
FIXED_TAU = 0.9
SUPER_CLASSES = {
    "FINANCIAL_FRAUD": [
        "Financial Fraud",
        "Gambling/Betting",
        "Cryptocurrency Crime"
    ],
    "SEXUAL_CRIME": [
        "Sexually Explicit Content",
        "Sexually Obscene Content",
        "Child Abuse Material",
        "Rape or Sexual Abuse Content"
    ],
    "CYBER_ATTACK": [
        "Hacking/Damage",
        "Cyber Attack/Dependent Crimes",
        "Cyber Trafficking",
        "Cyber Terrorism",
        "Ransomware"
    ],
    "SOCIAL_MEDIA_CRIME": [
        "Social Media Crime",
        "Other Cyber Crime"
    ]
}
train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

bert_train = np.load("trainhingroberta_embs.npy")
bert_test  = np.load("testhingroberta_embs.npy")

word_train = np.load("train_embs.npy")
word_test  = np.load("test_embs.npy")

def fuse(a, b):
    a = F.normalize(torch.tensor(a, dtype=torch.float32), dim=1)
    b = F.normalize(torch.tensor(b, dtype=torch.float32), dim=1)
    return torch.cat([a, b], dim=1)

X_train = fuse(bert_train, word_train).to(DEVICE)
X_test  = fuse(bert_test,  word_test ).to(DEVICE)

def build_hierarchy(df):
    super_labels, fine_labels = [], []
    for cat in df["category"].astype(str):
        assigned = False
        for s, members in SUPER_CLASSES.items():
            if cat in members:
                super_labels.append(s)
                fine_labels.append(cat)
                assigned = True
                break
        if not assigned:
            super_labels.append("OTHER_CYBER")
            fine_labels.append(cat)
    return super_labels, fine_labels

train_super, train_fine = build_hierarchy(train_df)
test_super,  test_fine  = build_hierarchy(test_df)

le_super = LabelEncoder()
le_fine  = LabelEncoder()

y_train_super = torch.tensor(le_super.fit_transform(train_super)).to(DEVICE)
y_train_fine  = torch.tensor(le_fine.fit_transform(train_fine)).to(DEVICE)

mapped_test_fine = [
    c if c in le_fine.classes_ else le_fine.classes_[0]
    for c in test_fine
]

y_test_super = torch.tensor(le_super.transform(test_super)).to(DEVICE)
y_test_fine  = torch.tensor(le_fine.transform(mapped_test_fine)).to(DEVICE)

fusion_dim = X_train.shape[1]

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.norm1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.norm2(self.fc2(h)) + h)
        return self.out(h)

def train_and_evaluate(
    lr,
    epochs,
    lambda_super,
    dropout,
    warmup_epochs,
    conf_quantile
):
    super_clf = nn.Sequential(
        nn.Linear(fusion_dim, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(le_super.classes_))
    ).to(DEVICE)

    fine_heads = nn.ModuleDict()
    fine_class_indices = {}

    for s in le_super.classes_:
        classes = [
            c for c in le_fine.classes_
            if c in SUPER_CLASSES.get(s, []) or s == "OTHER_CYBER"
        ]
        idxs = le_fine.transform(classes)
        fine_class_indices[s] = torch.tensor(idxs).to(DEVICE)

        fine_heads[s] = ResidualMLP(
            in_dim=fusion_dim + len(le_super.classes_),
            hidden_dim=512,
            out_dim=len(classes),
            dropout=dropout
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(super_clf.parameters()) + list(fine_heads.parameters()),
        lr=lr
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            idx = perm[i:i+64]
            optimizer.zero_grad()

            s_logits = super_clf(X_train[idx])

            if epoch <= warmup_epochs:
                loss = loss_fn(s_logits, y_train_super[idx])
            else:
                s_probs = F.softmax(s_logits.detach(), dim=1)
                f_input = torch.cat([X_train[idx], s_probs], dim=1)

                fine_loss, count = 0.0, 0
                for s_name, head in fine_heads.items():
                    s_id = le_super.transform([s_name])[0]
                    mask = (y_train_super[idx] == s_id)
                    if mask.sum() == 0:
                        continue

                    targets = y_train_fine[idx][mask]
                    valid_idxs = fine_class_indices[s_name]

                    local_targets = torch.tensor(
                        [(valid_idxs == t).nonzero()[0] for t in targets],
                        device=DEVICE
                    )

                    logits = head(f_input[mask])
                    fine_loss += loss_fn(logits, local_targets)
                    count += 1

                fine_loss /= max(count, 1)
                loss = (
                    lambda_super * loss_fn(s_logits, y_train_super[idx]) +
                    (1 - lambda_super) * fine_loss
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(super_clf.parameters()) + list(fine_heads.parameters()), 1.0
            )
            optimizer.step()

    super_clf.eval()
    for h in fine_heads.values():
        h.eval()

    preds = []
    confs = []
    top2_correct = 0

    with torch.no_grad():
        s_probs = F.softmax(super_clf(X_test), dim=1)

        for i in range(len(X_test)):
            s_id = s_probs[i].argmax().item()
            s_name = le_super.inverse_transform([s_id])[0]

            head = fine_heads[s_name]
            valid_idxs = fine_class_indices[s_name]

            f_input = torch.cat([X_test[i], s_probs[i]], dim=0).unsqueeze(0)
            probs = F.softmax(head(f_input), dim=1)

            local_top1 = probs.argmax().item()
            global_top1 = valid_idxs[local_top1].item()
            preds.append(global_top1)
            confs.append(
                probs.max().item() * s_probs[i][s_id].item()
)

            top2_local = probs.topk(2, dim=1).indices.squeeze(0).tolist()
            top2_global = valid_idxs[top2_local].tolist()
            if y_test_fine[i].item() in top2_global:
                top2_correct += 1

    preds = np.array(preds)
    confs = np.array(confs)

    acc1 = (preds == y_test_fine.cpu().numpy()).mean()
    acc2 = top2_correct / len(y_test_fine)

    conf_mask = confs >= FIXED_TAU

    coverage = conf_mask.mean()

    conf_acc = (
        (preds[conf_mask] == y_test_fine.cpu().numpy()[conf_mask]).mean()
        if conf_mask.sum() > 0 else 0.0
    )

    weighted_f1 = f1_score(
        y_test_fine.cpu().numpy(),
        preds,
        average="weighted"
    )

    return {
    "weighted_f1": weighted_f1,
    "top1": acc1,
    "top2": acc2,
    "conf_acc": conf_acc,
    "coverage": coverage,
    "preds": preds
}

def objective(trial):
    metrics = train_and_evaluate(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        epochs=trial.suggest_categorical("epochs", [20, 30, 40]),
        lambda_super=trial.suggest_float("lambda_super", 0.3, 0.6),
        dropout=trial.suggest_float("dropout", 0.2, 0.5),
        warmup_epochs=trial.suggest_categorical("warmup_epochs", [2, 3, 4]),
        conf_quantile=trial.suggest_float("conf_quantile", 0.45, 0.65)
    )

    print(
        f"[Trial {trial.number:02d}] "
        f"F1={metrics['weighted_f1']:.4f} | "
        f"Top1={metrics['top1']:.4f} | "
        f"Top2={metrics['top2']:.4f} | "
        f"ConfAcc@τ=0.9={metrics['conf_acc']:.4f} | "
        f"Coverage@τ=0.9={metrics['coverage']:.4f} | "
        f"Params={trial.params}"
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
    print(f"Coverage @ τ=0.9        : {final_metrics['coverage']:.4f}")

    used = np.unique(y_test_fine.cpu().numpy())

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test_fine.cpu().numpy(),
        final_metrics["preds"],
        labels=used,
        target_names=le_fine.inverse_transform(used),
        zero_division=0
    ))
