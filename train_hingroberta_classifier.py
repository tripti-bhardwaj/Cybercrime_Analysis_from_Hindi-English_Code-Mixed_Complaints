import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

MODEL_NAME = "l3cube-pune/hing-roberta"
DEVICE = "cpu"

EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 64
CONF_THRESH = 0.7

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

def clean_texts(texts):
    cleaned = []
    for t in texts:
        if t is None or pd.isna(t):
            cleaned.append("")
        else:
            t = str(t).strip()
            cleaned.append(t if len(t) > 0 else "")
    return cleaned

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
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_ds  = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=0
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
).to(DEVICE)

for param in model.roberta.embeddings.parameters():
    param.requires_grad = False

for layer in model.roberta.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

full_weights = np.ones(num_classes)
for c, w in zip(np.unique(train_labels), weights):
    full_weights[c] = w

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(full_weights, dtype=torch.float).to(DEVICE)
)

optimizer = AdamW(model.parameters(), lr=LR)

model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch} | Avg Loss {total_loss / len(train_loader):.4f}")

model.eval()

all_preds, all_labels = [], []
top2_hits, conf_correct, conf_total = 0, 0, 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch["labels"]

        logits = model(**batch).logits
        probs = F.softmax(logits, dim=1)

        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        top2 = torch.topk(probs, 2, dim=1).indices
        top2_hits += (top2 == labels.unsqueeze(1)).any(dim=1).sum().item()

        conf = probs.max(dim=1).values
        mask = conf >= CONF_THRESH
        conf_correct += (preds[mask] == labels[mask]).sum().item()
        conf_total += mask.sum().item()

top1 = accuracy_score(all_labels, all_preds)
top2 = top2_hits / len(all_labels)
conf_acc = conf_correct / conf_total if conf_total > 0 else 0

print("\nHingRoBERTa REPORT:")
labels = np.unique(all_labels)

print(classification_report(
    all_labels,
    all_preds,
    labels=labels,
    target_names=le.inverse_transform(labels),
    zero_division=0
))
print(f"\nTop-1 Accuracy       : {top1:.4f}")
print(f"Top-2 Accuracy       : {top2:.4f}")
print(f"Confidence Accuracy : {conf_acc:.4f}")
