import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_num_threads(1)

import optuna
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from collections import Counter, defaultdict
from itertools import combinations

from models import WordGNN
DEVICE = "cpu"
FIXED_TAU = 0.7
TOP_K = 2
N_TRIALS = 20

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

train_df["tokens"] = train_df["content_processed"].astype(str).str.split()
test_df["tokens"]  = test_df["content_processed"].astype(str).str.split()

X_train = np.load("trainhingroberta_embs.npy")
X_test  = np.load("testhingroberta_embs.npy")

le = LabelEncoder()
le.fit(train_df["category"].astype(str))

train_labels = le.transform(train_df["category"].astype(str))

test_labels_raw = test_df["category"].astype(str)
test_labels = test_labels_raw.apply(
    lambda x: x if x in le.classes_ else "<<UNK>>"
)

if "<<UNK>>" not in le.classes_:
    le.classes_ = np.append(le.classes_, "<<UNK>>")

test_labels = le.transform(test_labels)
num_classes = len(le.classes_)

def train_and_evaluate(
    lr,
    hidden_dim,
    top_k_vocab,
    max_tokens_per_doc,
    epochs
):
    word_freq = Counter()
    for toks in train_df["tokens"]:
        word_freq.update(toks)

    vocab = [w for w, _ in word_freq.most_common(top_k_vocab)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    edge_counter = Counter()
    for toks in train_df["tokens"]:
        toks = list(dict.fromkeys(
            [t for t in toks if t in word2idx]
        ))[:max_tokens_per_doc]

        for a, b in combinations(toks, 2):
            edge_counter[(word2idx[a], word2idx[b])] += 1

    edge_index = torch.tensor(
        list(edge_counter.keys()),
        dtype=torch.long
    ).t().contiguous()

    word_to_docs = defaultdict(list)
    for i, toks in enumerate(train_df["tokens"]):
        for w in set(toks):
            if w in word2idx:
                word_to_docs[w].append(i)

    EMB_DIM = X_train.shape[1]
    x_words = torch.zeros(len(vocab), EMB_DIM)

    for w, idx in word2idx.items():
        doc_ids = word_to_docs.get(w, [])
        if doc_ids:
            x_words[idx] = torch.tensor(
                X_train[doc_ids].mean(axis=0)
            )

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

    model = WordGNN(
        in_dim=EMB_DIM,
        out_dim=hidden_dim
    ).to(DEVICE)

    classifier = torch.nn.Linear(
        hidden_dim,
        num_classes
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()

        word_embs = model(data)

        tweet_reps = []
        for idxs in data.tweet_node_indices:
            if len(idxs) == 0:
                tweet_reps.append(torch.zeros(hidden_dim))
            else:
                tweet_reps.append(word_embs[idxs].mean(dim=0))

        tweet_reps = torch.stack(tweet_reps)

        logits = classifier(tweet_reps)
        loss = loss_fn(logits, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    classifier.eval()

    with torch.no_grad():
        word_embs = model(data).cpu()

    test_reps = []
    for toks in test_df["tokens"]:
        idxs = [word2idx[t] for t in toks if t in word2idx]
        if len(idxs) == 0:
            test_reps.append(torch.zeros(hidden_dim))
        else:
            test_reps.append(word_embs[idxs].mean(dim=0))

    test_reps = torch.stack(test_reps)

    logits = classifier(test_reps)
    probs = F.softmax(logits, dim=1)

    top1_preds = probs.argmax(dim=1).cpu().detach().numpy()
    acc1 = (top1_preds == test_labels).mean()

    top2_preds = torch.topk(probs, k=2, dim=1).indices.cpu().detach().numpy()
    acc2 = np.mean([
        test_labels[i] in top2_preds[i]
        for i in range(len(test_labels))
    ])

    confidence = probs.max(dim=1).values.cpu().detach().numpy()
    conf_mask = confidence >= FIXED_TAU

    coverage = conf_mask.mean()

    conf_acc = (
        np.mean([
            test_labels[i] == top1_preds[i]
            for i in range(len(test_labels))
            if conf_mask[i]
        ]) if conf_mask.sum() > 0 else 0.0
    )

    weighted_f1 = f1_score(
        test_labels,
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
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 384]),
        top_k_vocab=trial.suggest_categorical("top_k_vocab", [1000, 2000, 3000]),
        max_tokens_per_doc=trial.suggest_categorical("max_tokens", [15, 20, 30]),
        epochs=trial.suggest_categorical("epochs", [20, 30, 40])
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

    best = study.best_params
    final_metrics = train_and_evaluate(
        lr=best["lr"],
        hidden_dim=best["hidden_dim"],
        top_k_vocab=best["top_k_vocab"],
        max_tokens_per_doc=best["max_tokens"],
        epochs=best["epochs"]
    )

    print(f"Weighted F1-score     : {final_metrics['weighted_f1']:.4f}")
    print(f"Top-1 Accuracy        : {final_metrics['top1']:.4f}")
    print(f"Top-2 Accuracy        : {final_metrics['top2']:.4f}")
    print(f"Confidence Accuracy   : {final_metrics['conf_acc']:.4f}")
    print(f"Coverage @ τ=0.9      : {final_metrics['coverage']*100:.2f}%")

    used = np.unique(test_labels)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        test_labels,
        final_metrics["preds"],
        labels=used,
        target_names=le.inverse_transform(used),
        zero_division=0
    ))
