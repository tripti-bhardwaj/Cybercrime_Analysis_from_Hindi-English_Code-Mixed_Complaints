import torch

CONFIG = {
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "embedding_model": "all-MiniLM-L6-v2",
    "batch_size": 256,
    "ipc_k": 10,
    "epochs": 30,
    "lr": 1e-3,
}
