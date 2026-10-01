import numpy as np
import pandas as pd
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer


DEVICE = "cpu"
BATCH_SIZE = 64
LR = 3.58809011490994e-05
EPOCHS = 40
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 6
SUPER_TEMP = 0.9464270017695560
FINE_TEMP = 0.3871717192042970
CONF_QUANTILE = 0.55
DROPOUT_RATE = 0.4410404449009800
CONF_THRESHOLD = 0.9
MODEL_NAME = "l3cube-pune/hing-roberta"
MAX_LENGTH = 128
MODEL_PATH = "hierarchical_multihead_model.pt"

SUPER_CLASSES = {
    "FINANCIAL_FRAUD": [
        "Financial Fraud",
        "Gambling/Betting",
        "Cryptocurrency Crime",
    ],
    "SEXUAL_CRIME": [
        "Sexually Explicit Content",
        "Sexually Obscene Content",
        "Child Abuse Material",
        "Rape or Sexual Abuse Content",
    ],
    "CYBER_ATTACK": [
        "Hacking/Damage",
        "Cyber Attack/Dependent Crimes",
        "Cyber Trafficking",
        "Cyber Terrorism",
        "Ransomware",
    ],
    "SOCIAL_MEDIA_CRIME": ["Social Media Crime", "Other Cyber Crime"],
}
DEFAULT_FINE_CLASS = "Financial Fraud"


def build_hierarchy(df):
    super_labels, fine_labels = [], []
    for category in df["category"].astype(str):
        for super_class, members in SUPER_CLASSES.items():
            if category in members:
                super_labels.append(super_class)
                fine_labels.append(category)
                break
        else:
            super_labels.append("OTHER_CYBER")
            fine_labels.append(category)
    return super_labels, fine_labels


class AttentiveResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        h = self.drop(F.gelu(self.norm1(self.fc1(x))))
        h = h + F.gelu(self.norm2(self.fc2(h)))
        h = torch.sigmoid(self.attn(h)) * h
        return self.out(h)


def encode_complaints(texts, tokenizer, encoder, batch_size=BATCH_SIZE):
    """Encode raw complaint text using the same mean pooling as the embedding pipeline."""
    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = ["" if text is None else str(text).strip() for text in texts[start:start + batch_size]]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)
            output = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 768))


def _new_classifiers(fusion_dim, super_classes, fine_classes):
    super_clf = nn.Sequential(
        nn.Linear(fusion_dim, 768), nn.LayerNorm(768), nn.GELU(),
        nn.Dropout(DROPOUT_RATE), nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, len(super_classes)),
    ).to(DEVICE)
    fine_heads = nn.ModuleDict()
    fine_class_indices = {}
    for super_class in super_classes:
        classes = [
            category for category in fine_classes
            if category in SUPER_CLASSES.get(super_class, []) or super_class == "OTHER_CYBER"
        ]
        fine_class_indices[super_class] = torch.tensor(
            [list(fine_classes).index(category) for category in classes], device=DEVICE
        )
        fine_heads[super_class] = AttentiveResidualMLP(
            fusion_dim + len(super_classes), 512, len(classes)
        )
    return super_clf, fine_heads.to(DEVICE), fine_class_indices


def train_model(save_path=MODEL_PATH, evaluate=True):
    """Train the hierarchical classifier and save it for manual complaint prediction."""
    train_df = pd.read_csv("final_train.csv")
    test_df = pd.read_csv("final_test.csv")
    X_train = torch.tensor(np.load("trainhingroberta_embs.npy"), dtype=torch.float32, device=DEVICE)
    X_test = torch.tensor(np.load("testhingroberta_embs.npy"), dtype=torch.float32, device=DEVICE)
    fusion_dim = X_train.shape[1]

    train_super, train_fine = build_hierarchy(train_df)
    test_super, test_fine = build_hierarchy(test_df)
    le_super, le_fine = LabelEncoder(), LabelEncoder()
    y_train_super = torch.tensor(le_super.fit_transform(train_super), device=DEVICE)
    y_train_fine = torch.tensor(le_fine.fit_transform(train_fine), device=DEVICE)

    mapped_test_fine = [
        category if category in le_fine.classes_ else DEFAULT_FINE_CLASS
        for category in test_fine
    ]
    y_test_fine = torch.tensor(le_fine.transform(mapped_test_fine), device=DEVICE)

    fine_counts = np.bincount(y_train_fine.cpu().numpy())
    class_weights = torch.tensor(
        1.0 / np.sqrt(fine_counts + 1), dtype=torch.float32, device=DEVICE
    )

    super_clf, fine_heads, fine_class_indices = _new_classifiers(
        fusion_dim, le_super.classes_, le_fine.classes_
    )

    optimizer = torch.optim.AdamW(
        list(super_clf.parameters()) + list(fine_heads.parameters()), lr=LR
    )
    super_loss_fn = nn.CrossEntropyLoss()

    # Stage 1 trains the superclass classifier; stage 2 then trains both levels.
    stage2_start = STAGE1_EPOCHS + 1
    stage2_end = min(stage2_start + STAGE2_EPOCHS - 1, EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        super_clf.train()
        fine_heads.train()
        train_fine_heads = stage2_start <= epoch <= stage2_end
        for parameter in fine_heads.parameters():
            parameter.requires_grad = train_fine_heads

        permutation = torch.randperm(len(X_train))
        total_loss = 0.0
        for start in range(0, len(X_train), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            super_logits = super_clf(X_train[indices])
            loss_super = super_loss_fn(super_logits, y_train_super[indices])
            loss = loss_super

            if train_fine_heads:
                super_probs = F.softmax(super_logits.detach() / SUPER_TEMP, dim=1)
                fine_input = torch.cat([X_train[indices], super_probs], dim=1)
                fine_loss, used_heads = 0.0, 0
                for super_class, head in fine_heads.items():
                    super_id = le_super.transform([super_class])[0]
                    mask = y_train_super[indices] == super_id
                    if not mask.any():
                        continue

                    valid = fine_class_indices[super_class]
                    targets = y_train_fine[indices][mask]
                    local_targets = torch.stack(
                        [(valid == target).nonzero(as_tuple=True)[0][0] for target in targets]
                    )
                    logits = head(fine_input[mask]) / FINE_TEMP
                    ce = nn.CrossEntropyLoss(weight=class_weights[valid])(
                        logits, local_targets
                    )
                    if logits.shape[1] > 1:
                        top2 = logits.topk(2, dim=1).values
                        margin = torch.clamp(0.25 - (top2[:, 0] - top2[:, 1]), min=0).mean()
                    else:
                        margin = logits.new_zeros(())
                    fine_loss = fine_loss + ce + 0.3 * margin
                    used_heads += 1

                if used_heads:
                    loss = 0.4 * loss_super + 0.6 * (fine_loss / used_heads)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(super_clf.parameters()) + list(fine_heads.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        stage_name = "superclass" if epoch < stage2_start else (
            "fine-tuning" if epoch <= stage2_end else "superclass"
        )
        print(f"Epoch {epoch:02d} ({stage_name}) | Avg Loss {total_loss / len(X_train):.4f}")

    torch.save({
        "super_state": super_clf.state_dict(),
        "fine_state": fine_heads.state_dict(),
        "super_classes": list(le_super.classes_),
        "fine_classes": list(le_fine.classes_),
        "fine_class_indices": {key: value.cpu() for key, value in fine_class_indices.items()},
        "fusion_dim": fusion_dim,
        "model_name": MODEL_NAME,
    }, save_path)
    print(f"Saved complaint predictor to {save_path}")

    if not evaluate:
        return save_path

    super_clf.eval()
    fine_heads.eval()
    predictions, confidences, top2_correct = [], [], 0
    with torch.no_grad():
        super_probs = F.softmax(super_clf(X_test) / SUPER_TEMP, dim=1)
        for i in range(len(X_test)):
            scores = {}
            super_scores, super_ids = super_probs[i].topk(2)
            for rank, super_id in enumerate(super_ids):
                super_class = le_super.inverse_transform([super_id.item()])[0]
                valid = fine_class_indices[super_class]
                model_input = torch.cat([X_test[i], super_probs[i]]).unsqueeze(0)
                fine_probs = F.softmax(fine_heads[super_class](model_input) / FINE_TEMP, dim=1)[0]
                local_top2 = fine_probs.topk(min(2, fine_probs.numel())).indices
                if y_test_fine[i] in valid[local_top2]:
                    top2_correct += 1
                predicted_class = valid[fine_probs.argmax()].item()
                scores[predicted_class] = scores.get(predicted_class, 0.0) + (
                    super_scores[rank].item() * fine_probs.max().item()
                )
            prediction, confidence = max(scores.items(), key=lambda item: item[1])
            predictions.append(prediction)
            confidences.append(confidence)

    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)
    truth = y_test_fine.cpu().numpy()
    confidence_threshold = max(
        CONF_THRESHOLD, float(np.quantile(confidences, CONF_QUANTILE))
    )
    confident = confidences >= confidence_threshold
    coverage = confident.mean()
    confidence_accuracy = (predictions[confident] == truth[confident]).mean() if confident.any() else 0.0
    weighted_f1 = f1_score(truth, predictions, average="weighted", zero_division=0)

    print("\nFINAL METRICS")
    print(f"Weighted F1-score     : {weighted_f1:.4f}")
    print(f"Top-1 Accuracy        : {(predictions == truth).mean():.4f}")
    print(f"Top-2 Accuracy        : {top2_correct / len(truth):.4f}")
    print(f"Confidence Accuracy   : {confidence_accuracy:.4f}")
    print(f"Coverage              : {coverage:.4f}")
    print(f"Confidence Threshold  : {confidence_threshold:.4f}")
    print("\nCLASSIFICATION REPORT:")
    labels = np.unique(truth)
    print(classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=le_fine.inverse_transform(labels),
        zero_division=0,
    ))
    return save_path


@lru_cache(maxsize=2)
def _load_predictor(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    encoder = AutoModel.from_pretrained(checkpoint["model_name"]).to(DEVICE)
    super_classes = checkpoint["super_classes"]
    fine_classes = checkpoint["fine_classes"]
    super_clf, fine_heads, fine_class_indices = _new_classifiers(
        checkpoint["fusion_dim"], super_classes, fine_classes
    )
    super_clf.load_state_dict(checkpoint["super_state"])
    fine_heads.load_state_dict(checkpoint["fine_state"])
    super_clf.eval()
    fine_heads.eval()
    return checkpoint, tokenizer, encoder, super_clf, fine_heads, fine_class_indices


def predict_complaint(complaint, model_path=MODEL_PATH, top_k=3):
    """Return category, superclass, confidence, and alternatives for one raw complaint."""
    checkpoint, tokenizer, encoder, super_clf, fine_heads, fine_class_indices = _load_predictor(model_path)
    super_classes = checkpoint["super_classes"]
    fine_classes = checkpoint["fine_classes"]

    embedding = encode_complaints([complaint], tokenizer, encoder)
    with torch.no_grad():
        super_probs = F.softmax(super_clf(embedding) / SUPER_TEMP, dim=1)[0]
        candidate_scores = {}
        for super_prob, super_id in zip(*super_probs.topk(min(2, len(super_classes)))):
            super_class = super_classes[super_id.item()]
            valid = fine_class_indices[super_class]
            model_input = torch.cat([embedding[0], super_probs]).unsqueeze(0)
            fine_probs = F.softmax(fine_heads[super_class](model_input) / FINE_TEMP, dim=1)[0]
            for local_id, probability in enumerate(fine_probs):
                category = fine_classes[valid[local_id].item()]
                candidate_scores[category] = candidate_scores.get(category, 0.0) + (
                    super_prob.item() * probability.item()
                )

    alternatives = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    category, confidence = alternatives[0]
    super_class = next(
        (name for name, members in SUPER_CLASSES.items() if category in members),
        "OTHER_CYBER",
    )
    return {
        "complaint": complaint,
        "category": category,
        "superclass": super_class,
        "confidence": confidence,
        "meets_confidence_threshold": confidence >= CONF_THRESHOLD,
        "alternatives": [{"category": name, "confidence": score} for name, score in alternatives],
    }


if __name__ == "__main__":
    train_model()
