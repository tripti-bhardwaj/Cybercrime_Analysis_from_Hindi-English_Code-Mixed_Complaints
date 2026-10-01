import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

DEVICE = "cpu"

EPOCHS = 30
BATCH = 64
LR = 3e-4

TOP_K = 2
CONF_QUANTILE = 0.55
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

train_fine_set = set(le_fine.classes_)
mapped_test_fine = [
    c if c in train_fine_set else "any other cyber crime"
    for c in test_fine
]

y_test_super = torch.tensor(le_super.transform(test_super)).to(DEVICE)
y_test_fine  = torch.tensor(le_fine.transform(mapped_test_fine)).to(DEVICE)
fusion_dim = X_train.shape[1]

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.35):
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

super_clf = nn.Sequential(
    nn.Linear(fusion_dim, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.3),
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
        out_dim=len(classes)
    ).to(DEVICE)
loss_fn_super = nn.CrossEntropyLoss()
loss_fn_fine  = nn.CrossEntropyLoss()

LAMBDA_SUPER = 0.4
LAMBDA_FINE  = 0.6

optimizer = torch.optim.AdamW(
    list(super_clf.parameters()) + list(fine_heads.parameters()),
    lr=LR
)

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(X_train))
    total_loss = 0.0

    if epoch <= 3:
        for p in fine_heads.parameters():
            p.requires_grad = False
    else:
        for p in fine_heads.parameters():
            p.requires_grad = True

    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]
        optimizer.zero_grad()

        s_logits = super_clf(X_train[idx])

        if epoch <= 3:
            loss = loss_fn_super(s_logits, y_train_super[idx])
        else:
            s_probs = F.softmax(s_logits.detach(), dim=1)
            f_input = torch.cat([X_train[idx], s_probs], dim=1)

            fine_loss = 0.0
            count = 0

            for s_name, head in fine_heads.items():
                mask = (y_train_super[idx] == le_super.transform([s_name])[0])
                if mask.sum() == 0:
                    continue

                f_batch = f_input[mask]
                targets = y_train_fine[idx][mask]

                valid_idxs = fine_class_indices[s_name]

                local_targets = torch.tensor([
                    (valid_idxs == t).nonzero(as_tuple=True)[0].item()
                    for t in targets
                ], device=DEVICE)

                logits = head(f_batch)
                fine_loss += loss_fn_fine(logits, local_targets)
                count += 1

            fine_loss /= max(count, 1)
            loss = (
                LAMBDA_SUPER * loss_fn_super(s_logits, y_train_super[idx]) +
                LAMBDA_FINE  * fine_loss
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(super_clf.parameters()) + list(fine_heads.parameters()),
            1.0
        )
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Avg Loss {(total_loss/len(X_train)):.4f}")

super_clf.eval()
for h in fine_heads.values():
    h.eval()

top1_preds = []
top2_correct = 0
top1_conf = []

with torch.no_grad():
    s_logits = super_clf(X_test)
    s_probs  = F.softmax(s_logits, dim=1)

    for i in range(len(X_test)):
        s_id = s_probs[i].argmax().item()
        s_name = le_super.inverse_transform([s_id])[0]

        head = fine_heads[s_name]
        valid_idxs = fine_class_indices[s_name]

        f_input = torch.cat([X_test[i], s_probs[i]], dim=0).unsqueeze(0)
        logits = head(f_input)
        probs = F.softmax(logits, dim=1)

        conf = probs.max().item()
        top1_conf.append(conf)

        local_top1 = probs.argmax(dim=1).item()
        global_top1 = valid_idxs[local_top1].item()
        top1_preds.append(global_top1)

        top2_local = probs.topk(2, dim=1).indices.squeeze(0).tolist()
        top2_global = valid_idxs[top2_local].tolist()
        if y_test_fine[i].item() in top2_global:
            top2_correct += 1

top1_preds = torch.tensor(top1_preds).to(DEVICE)
top1_conf = torch.tensor(top1_conf).to(DEVICE)

top1_acc = (top1_preds == y_test_fine).float().mean().item()
top2_acc = top2_correct / len(y_test_fine)

CONF_THRESH = top1_conf.quantile(CONF_QUANTILE).item()
confident = top1_conf >= CONF_THRESH

confidence_acc = (
    (top1_preds[confident] == y_test_fine[confident])
    .float().mean().item()
)

print("\nHIERARCHICAL REPORT")
print(classification_report(
    y_test_fine.cpu().numpy(),
    top1_preds.cpu().numpy(),
    target_names=le_fine.classes_,
    zero_division=0
))

print(f"\nTop-1 Accuracy        : {top1_acc:.4f}")
print(f"Top-2 Accuracy        : {top2_acc:.4f}")
print(f"Confidence Accuracy   : {confidence_acc:.4f}")
