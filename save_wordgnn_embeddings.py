import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import Counter
from tqdm import tqdm

from models import WordGNN
from config import CONFIG

DEVICE = CONFIG["device"]

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

train_df["tokens"] = train_df["content_processed"].astype(str).str.split()
test_df["tokens"]  = test_df["content_processed"].astype(str).str.split()

word_freq = Counter()
for toks in train_df["tokens"]:
    word_freq.update(toks)

TOP_K = 2000
vocab = [w for w, _ in word_freq.most_common(TOP_K)]
word2idx = {w: i for i, w in enumerate(vocab)}

edges = Counter()
WINDOW = 4
MIN_EDGE_FREQ = 5

for toks in tqdm(train_df["tokens"], desc="Building WCG"):
    toks = [t for t in toks if t in word2idx]
    for i in range(len(toks)):
        for j in range(i + 1, min(i + WINDOW, len(toks))):
            a, b = word2idx[toks[i]], word2idx[toks[j]]
            if a != b:
                edges[(a, b)] += 1
                edges[(b, a)] += 1

filtered_edges = [(i, j) for (i, j), c in edges.items() if c >= MIN_EDGE_FREQ]

edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

print(f"WCG edges: {edge_index.size(1)}")

EMB_DIM = 128
x_words = torch.randn(len(vocab), EMB_DIM)

data = Data(
    x=x_words.to(DEVICE),
    edge_index=edge_index.to(DEVICE)
)

model = WordGNN(in_dim=EMB_DIM, out_dim=256).to(DEVICE)
model.eval()

with torch.no_grad():
    word_embs = model(data).cpu()

def pool(df):
    reps = []
    for toks in df["tokens"]:
        idxs = [word2idx[t] for t in toks if t in word2idx]
        if len(idxs) == 0:
            reps.append(torch.zeros(256))
        else:
            reps.append(word_embs[idxs].mean(dim=0))
    return torch.stack(reps)

train_embs = pool(train_df)
test_embs  = pool(test_df)

torch.save(edge_index.cpu(), "wcg_edge_index.pt")
np.save("train_embs.npy", train_embs.numpy())
np.save("test_embs.npy", test_embs.numpy())

print("Saved embeddings:")
print("Train:", train_embs.shape)
print("Test :", test_embs.shape)
