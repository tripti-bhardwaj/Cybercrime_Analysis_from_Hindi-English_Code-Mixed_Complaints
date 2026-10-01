import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ita_mapping_and_preprocessing import map_ita_sections

ALPHA = 0.6
TOP_K = 3
K_NEIGHBORS = 10
MODE = "hybrid"  # options: "keyword", "graph", "hybrid"

train_df = pd.read_csv("final_train.csv")
test_df  = pd.read_csv("final_test.csv")

X_train = np.load("trainhingroberta_embs.npy")
X_test  = np.load("testhingroberta_embs.npy")

assert X_train.shape[1] == X_test.shape[1], \
    "Embedding dimension mismatch between train and test"

def graph_confidence(test_emb: np.ndarray, train_embs: np.ndarray, k: int = 10) -> float:
    sims = cosine_similarity(test_emb.reshape(1, -1), train_embs)[0]
    topk = np.sort(sims)[-k:]
    return float(np.mean(topk))

final_results = []

for i, text in tqdm(
    enumerate(test_df["content_processed"].astype(str)),
    total=len(test_df),
    desc=f"ITA Prediction ({MODE})"
):
    keyword_matches = map_ita_sections(text)
    gconf = graph_confidence(X_test[i], X_train, K_NEIGHBORS)

    predictions = []

    if MODE == "keyword":
        for m in keyword_matches:
            predictions.append({
                "it_act_section": m["it_act_section"],
                "description": m["description"],
                "confidence": round(m["keyword_confidence"], 3),
                "keyword_hits": m["keyword_hits"],
                "graph_support": 0.0
            })

    elif MODE == "graph":
        # Assign same graph score to all candidate sections
        for m in keyword_matches:
            predictions.append({
                "it_act_section": m["it_act_section"],
                "description": m["description"],
                "confidence": round(gconf, 3),
                "keyword_hits": 0,
                "graph_support": round(gconf, 3)
            })

        # fallback if no keyword matches
        if not predictions:
            predictions.append({
                "it_act_section": "43",
                "description": "Unauthorized access / damage",
                "confidence": round(gconf, 3),
                "keyword_hits": 0,
                "graph_support": round(gconf, 3)
            })

    else:
        for m in keyword_matches:
            combined_conf = ALPHA * m["keyword_confidence"] + (1 - ALPHA) * gconf
            predictions.append({
                "it_act_section": m["it_act_section"],
                "description": m["description"],
                "confidence": round(combined_conf, 3),
                "keyword_hits": m["keyword_hits"],
                "graph_support": round(gconf, 3)
            })

        if not predictions:
            predictions.append({
                "it_act_section": "43",
                "description": "Unauthorized access / damage",
                "confidence": round(gconf, 3),
                "keyword_hits": 0,
                "graph_support": round(gconf, 3)
            })

    # Sort and take top-K
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[:TOP_K]

    final_results.append({
        "text": text,
        "ita_predictions": predictions
    })

final_df = pd.DataFrame(final_results)
output_file = f"ita_predictions_{MODE}.csv"
final_df.to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")
