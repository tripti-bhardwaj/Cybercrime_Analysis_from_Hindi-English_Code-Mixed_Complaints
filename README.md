# Cybercrime Text Classification

Source code for a research project on classifying cybercrime complaints using hierarchical and multi-head neural classifiers, Hinglish transformer representations, feature fusion, and graph-based methods. Scripts ending in `_hpt.py` run hyperparameter tuning; scripts beginning with `train_` train model variants.

## Repository contents

- `train_*.py`: model training scripts.
- `*_hpt.py`: Optuna hyperparameter-tuning scripts.
- `precompute_hingroberta_embeddings.py`: creates HingRoBERTa text embeddings.
- `train_hierarchical_multihead_classifier.py`: hierarchical multi-head training and manual complaint prediction functions.
- `ita_mapping*.py`, `evaluate_ita_mapping.py`: IT Act section mapping and evaluation utilities.
- `requirements.txt`: Python dependencies.

Dataset files, precomputed embeddings, and model checkpoints are excluded. Obtain the dataset through its authorized source and place the expected files in the project directory before running scripts. Training scripts expect files such as `final_train.csv`, `final_test.csv`, and, where applicable, precomputed embedding arrays. Some scripts download pretrained models from Hugging Face on first use.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and try a complaint

From the repository directory, run:

```bash
python train_hierarchical_multihead_classifier.py
```

The script trains the hierarchical model and writes `hierarchical_multihead_model.pt` locally (the checkpoint is excluded from Git). To predict a manual complaint after training:

```python
from train_hierarchical_multihead_classifier import predict_complaint

result = predict_complaint(
    "Someone called pretending to be my bank and took money from my account."
)
print(result)
```

The first prediction downloads/loads the HingRoBERTa model; subsequent predictions in the same Python process reuse it.

## Reproducibility notes

Each training and tuning script may have its own expected input files and configuration. Review the constants near the top of a script before running it. Hyperparameter-tuning scripts may run many trials and take substantial time.

## Citation

If you use this code in a paper, cite the associated paper and include this repository URL. Add the final paper citation here once its bibliographic details are available.
