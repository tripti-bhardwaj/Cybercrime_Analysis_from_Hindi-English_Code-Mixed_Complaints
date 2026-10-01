import pandas as pd
import ast
from tqdm import tqdm

MODE = "hybrid"  # options: "keyword", "graph", "hybrid"

# Load files
pred_df = pd.read_csv(f"ita_predictions_{MODE}.csv")
gt_df   = pd.read_csv("final_test.csv")
print("Ground Truth Columns:", gt_df.columns.tolist())

# Merge predictions with ground truth
# Ground truth column is 'category' (actual IT Act section label)
gt_col = "category"

df = pd.concat([pred_df, gt_df[gt_col]], axis=1)
df.rename(columns={gt_col: "it_act_section"}, inplace=True)

method_name = {
    "keyword": "Keyword-Based",
    "graph": "Graph-Based",
    "hybrid": "LexiGraph (Hybrid)"
}[MODE]

# Metrics counters
top1_correct = 0
topk_correct = 0
mrr_total = 0
coverage = 0
precision_total = 0

K = len(ast.literal_eval(pred_df.iloc[0]["ita_predictions"]))

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):

    true_label = str(row["it_act_section"]).strip()

    # Convert string → list of dicts
    preds = ast.literal_eval(row["ita_predictions"])

    pred_sections = [str(p["it_act_section"]) for p in preds]

    # ✅ Top-1 Accuracy
    if len(pred_sections) > 0 and pred_sections[0] == true_label:
        top1_correct += 1

    # ✅ Top-K Recall
    if true_label in pred_sections:
        topk_correct += 1

        # ✅ MRR
        rank = pred_sections.index(true_label) + 1
        mrr_total += 1 / rank

    # Precision@K
    correct_preds = sum([1 for p in pred_sections if p == true_label])
    precision_total += correct_preds / len(pred_sections) if pred_sections else 0

    # ✅ Coverage (at least 1 prediction exists)
    if len(pred_sections) > 0:
        coverage += 1

# Final metrics
n = len(df)

top1_acc = top1_correct / n
topk_recall = topk_correct / n
mrr = mrr_total / n
coverage_score = coverage / n
precision_at_k = precision_total / n

# Print results
print("\n===== LexiGraph ITA Mapping Evaluation =====")
print(f"Top-1 Accuracy     : {top1_acc:.4f}")
print(f"Top-{K} Recall     : {topk_recall:.4f}")
print(f"MRR                : {mrr:.4f}")
print(f"Coverage           : {coverage_score:.4f}")
print(f"Precision@{K}       : {precision_at_k:.4f}")

errors = []

for _, row in df.iterrows():
    true_label = str(row["it_act_section"]).strip()
    preds = ast.literal_eval(row["ita_predictions"])
    pred_sections = [str(p["it_act_section"]) for p in preds]

    if true_label not in pred_sections:
        errors.append({
            "text": row["text"],
            "true": true_label,
            "predicted": pred_sections,
            "top_prediction": pred_sections[0] if pred_sections else None
        })

error_df = pd.DataFrame(errors)
error_df.to_csv("ita_errors.csv", index=False)

print(f"Total Errors: {len(errors)}")

# Create summary table for LexiGraph
results_table = pd.DataFrame({
    "Method": [method_name],
    "Top-1 Accuracy": [round(top1_acc, 4)],
    "Top-3 Recall": [round(topk_recall, 4)],
    "MRR": [round(mrr, 4)],
    "Coverage": [round(coverage_score, 4)],
    "Precision@K": [round(precision_at_k, 4)],
})

# Save as CSV
results_table.to_csv(f"{MODE}_results_table.csv", index=False)

print("\n📊 LexiGraph Results Table:")
print(results_table)


# =========================
# 🔬 Ablation Comparison Table (Manual Entry Required)
# =========================

# Fill these values AFTER running keyword-only and graph-only models
ablation_table = pd.DataFrame({
    "Method": [
        "Keyword-Based",
        "Graph-Based",
        "LexiGraph (Hybrid)"
    ],
    "Top-1 Accuracy": [
        None,   # Fill manually
        None,   # Fill manually
        round(top1_acc, 4)
    ],
    "Top-3 Recall": [
        None,
        None,
        round(topk_recall, 4)
    ],
    "MRR": [
        None,
        None,
        round(mrr, 4)
    ],
    "Coverage": [
        None,
        None,
        round(coverage_score, 4)
    ]
})

# Save ablation table
ablation_table.to_csv("ablation_comparison_table.csv", index=False)

print("\n🔬 Ablation Comparison Table (fill missing values):")
print(ablation_table)
