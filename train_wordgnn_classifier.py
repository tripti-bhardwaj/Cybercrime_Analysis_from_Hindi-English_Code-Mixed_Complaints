import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from collections import Counter, defaultdict
from itertools import combinations
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from models import WordGNN
from config import CONFIG

DEVICE = "cpu"
train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

train_df["tokens"] = train_df["content_processed"].astype(str).str.split()
test_df["tokens"]  = test_df["content_processed"].astype(str).str.split()

X_train = np.load("trainhingroberta_embs.npy")
X_test  = np.load("testhingroberta_embs.npy")

le = LabelEncoder()
le.fit(train_df["category"].astype(str))

train_labels = le.transform(train_df["category"].astype(str))

known = set(le.classes_)
test_labels_raw = test_df["category"].astype(str)
test_labels_mapped = test_labels_raw.apply(
    lambda x: x if x in known else "<<UNK>>"
)

if "<<UNK>>" in test_labels_mapped.values:
    le.classes_ = np.append(le.classes_, "<<UNK>>")

test_labels = le.transform(test_labels_mapped)
num_classes = len(le.classes_)

word_freq = Counter()
for toks in train_df["tokens"]:
    word_freq.update(toks)

TOP_K = 2000
vocab = [w for w, _ in word_freq.most_common(TOP_K)]
word2idx = {w: i for i, w in enumerate(vocab)}

edge_counter = Counter()
MAX_TOKENS_PER_DOC = 20

for toks in train_df["tokens"]:
    toks = list(dict.fromkeys([t for t in toks if t in vocab]))[:MAX_TOKENS_PER_DOC]
    for a, b in combinations(toks, 2):
        edge_counter[(word2idx[a], word2idx[b])] += 1

edge_index = torch.tensor(list(edge_counter.keys())).t().contiguous()

word_to_tweets = defaultdict(list)
for i, toks in enumerate(train_df["tokens"]):
    for w in set(toks):
        if w in word2idx:
            word_to_tweets[w].append(i)

EMB_DIM = X_train.shape[1]
x_words = torch.zeros(len(vocab), EMB_DIM)

for w, idx in word2idx.items():
    tweet_ids = word_to_tweets.get(w, [])
    if tweet_ids:
        x_words[idx] = torch.tensor(X_train[tweet_ids].mean(axis=0))

tweet_word_indices = []
for toks in train_df["tokens"]:
    idxs = [word2idx[t] for t in toks if t in word2idx]
    tweet_word_indices.append(torch.tensor(idxs))

data = Data(
    x=x_words,
    edge_index=edge_index,
    y=torch.tensor(train_labels)
)
data.tweet_node_indices = tweet_word_indices
data = data.to(DEVICE)

model = WordGNN(in_dim=EMB_DIM, out_dim=256).to(DEVICE)
classifier = torch.nn.Linear(256, num_classes).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=CONFIG["lr"]
)

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(CONFIG["epochs"]):
    model.train()
    classifier.train()
    optimizer.zero_grad()

    word_embs = model(data)

    tweet_reps = []
    for idxs in data.tweet_node_indices:
        if len(idxs) == 0:
            tweet_reps.append(torch.zeros(256))
        else:
            tweet_reps.append(word_embs[idxs].mean(dim=0))

    tweet_reps = torch.stack(tweet_reps)

    logits = classifier(tweet_reps)
    loss = loss_fn(logits, data.y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1:02d} | Loss={loss.item():.4f}")

model = model.to("cpu")
classifier = classifier.to("cpu")
data = data.to("cpu")

model.eval()
classifier.eval()

with torch.no_grad():
    word_embs = model(data).cpu()
test_reps = []
for toks in test_df["tokens"]:
    idxs = [word2idx[t] for t in toks if t in word2idx]
    if len(idxs) == 0:
        test_reps.append(torch.zeros(256))
    else:
        test_reps.append(word_embs[idxs].mean(dim=0))

test_reps = torch.stack(test_reps)

test_logits = classifier(test_reps)
probs = F.softmax(test_logits, dim=1)

top1_preds = probs.argmax(dim=1)
top1_acc = (top1_preds.numpy() == test_labels).mean()

K = 2
topk_probs, topk_preds = torch.topk(probs, k=K, dim=1)
top2_acc = np.mean([
    test_labels[i] in topk_preds[i].numpy()
    for i in range(len(test_labels))
])

CONF_THRESH = 0.7
max_conf = topk_probs[:, 0]
conf_mask = max_conf >= CONF_THRESH

confidence_acc = (
    np.mean([
        test_labels[i] in topk_preds[i].numpy()
        for i in range(len(test_labels)) if conf_mask[i]
    ]) if conf_mask.sum() > 0 else 0
)

used_labels = np.unique(test_labels)
used_target_names = le.inverse_transform(used_labels)

print("\nWordGNN REPORT:")
print(
    classification_report(
        test_labels,
        top1_preds.numpy(),
        labels=used_labels,
        target_names=used_target_names,
        zero_division=0
    )
)

print(f"\nTop-1 Accuracy      : {top1_acc:.4f}")
print(f"Top-2 Accuracy        : {top2_acc:.4f}")
print(f"Confidence Accuracy   : {confidence_acc:.4f}")
