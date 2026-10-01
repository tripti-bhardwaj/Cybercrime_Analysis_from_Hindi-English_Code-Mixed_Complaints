import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

TRAIN_PATH = "final_train.csv"
TEST_PATH  = "final_test.csv"

TRAIN_SAVE = "trainhingroberta_embs.npy"
TEST_SAVE  = "testhingroberta_embs.npy"

MODEL_NAME = "l3cube-pune/hing-roberta"
BATCH_SIZE = 64
MAX_LEN = 128

from config import CONFIG
DEVICE = CONFIG["device"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def encode(texts):
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
            batch = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embs.append(embeddings.cpu().numpy())

    return np.vstack(all_embs)

train_df = pd.read_csv(TRAIN_PATH)
train_texts = train_df["content_processed"].astype(str).tolist()

train_embs = encode(train_texts)
np.save(TRAIN_SAVE, train_embs)

test_df = pd.read_csv(TEST_PATH)
test_texts = test_df["content_processed"].astype(str).tolist()

test_embs = encode(test_texts)
np.save(TEST_SAVE, test_embs)

print("Saved HingRoBERTa embeddings")
print("Train:", train_embs.shape)
print("Test :", test_embs.shape)
