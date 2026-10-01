import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

DEVICE = "cpu"

EPOCHS = 60
BATCH = 128
LR = 3e-4
TEMPERATURE = 0.7
TOP_K = 2
CONF_QUANTILE = 0.55

SEED = 1337 #42/1337
MODEL_PATH = f"fusion_run{SEED}.pt"

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

train_labels = set(le.classes_)
mapped_test = [
    c if c in train_labels else le.classes_[0]
    for c in test_df["category"].astype(str)
]

y_test = torch.tensor(le.transform(mapped_test)).to(DEVICE)

NUM_CLASSES = len(le.classes_)
FUSION_DIM = X_train.shape[1]
class FlatFusionClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = FlatFusionClassifier(FUSION_DIM, NUM_CLASSES).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.15)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(X_train))
    total_loss = 0.0

    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]

        optimizer.zero_grad()
        logits = model(X_train[idx])
        loss = loss_fn(logits, y_train[idx])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"[Seed {SEED}] Epoch {epoch:02d} | Avg Loss {(total_loss/len(X_train)):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
if SEED == 1337:
    print("\n Running Ensemble Evaluation")

    model1 = FlatFusionClassifier(FUSION_DIM, NUM_CLASSES).to(DEVICE)
    model2 = FlatFusionClassifier(FUSION_DIM, NUM_CLASSES).to(DEVICE)

    model1.load_state_dict(torch.load("fusion_run42.pt"))
    model2.load_state_dict(torch.load("fusion_run1337.pt"))

    model1.eval()
    model2.eval()

    with torch.no_grad():
        logits1 = model1(X_test)
        logits2 = model2(X_test)

        probs1 = F.softmax(logits1 / TEMPERATURE, dim=1)
        probs2 = F.softmax(logits2 / TEMPERATURE, dim=1)

        probs = (probs1 + probs2) / 2

    topk_probs, topk_preds = torch.topk(probs, k=TOP_K, dim=1)
    top1_preds = topk_preds[:, 0]

    top1_acc = (top1_preds == y_test).float().mean().item()
    top2_acc = sum(
        y_test[i].item() in topk_preds[i].tolist()
        for i in range(len(y_test))
    ) / len(y_test)

    sorted_probs, _ = torch.sort(probs, descending=True, dim=1)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    CONF_THRESH = margin.quantile(CONF_QUANTILE).item()
    confident = margin >= CONF_THRESH

    confidence_acc = (
        (top1_preds[confident] == y_test[confident])
        .float().mean().item()
    )
    labels_present = np.unique(y_test.cpu().numpy())

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test.cpu().numpy(),
        top1_preds.cpu().numpy(),
        labels=labels_present,
        target_names=[le.classes_[i] for i in labels_present],
        zero_division=0
    ))

    print(f"\nENSEMBLE RESULTS")
    print(f"Top-1 Accuracy        : {top1_acc:.4f}")
    print(f"Top-2 Accuracy        : {top2_acc:.4f}")
    print(f"Confidence Accuracy   : {confidence_acc:.4f}")
